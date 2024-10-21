from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput, has_length
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.deepspeed import deepspeed_init
from transformers.trainer import logger
import datasets
import torch

def safe_str(obj):
    try: return str(obj)
    except UnicodeEncodeError:
        return obj.encode('ascii', 'ignore').decode('ascii')
    return ""

class MyTrainer(Trainer):

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on CPU (final containers)
        all_preds = []
        all_labels = []
        all_inputs = []

        observed_num_examples = 0
        # Main evaluation loop
        for step, batch in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = len(batch)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            encoding = self.tokenizer(batch["x"], padding=True, return_tensors='pt')
            encoding = self._prepare_inputs(encoding)
            max_length = min(self.tokenizer.model_max_length, encoding['input_ids'].size(1) + 512)
            generated_ids = self.model.generate(**encoding, max_length=max_length)
            try:
                generated_texts = self.tokenizer.batch_decode(
                    generated_ids[:, encoding['input_ids'].size(1):], 
                    skip_special_tokens=True)
            except:
                print("cannot decode: ")
                print(generated_ids)

            all_preds.extend(generated_texts)
            all_labels.extend(batch["y"])
            all_inputs.extend(batch["x"])

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)


        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        metrics = {"acc": 0.0}
        num_correct = 0
        num_all = 0
        outputs = []
        assert len(all_preds) == len(all_labels)
        assert len(all_preds) == len(all_inputs)
        for pred, x, y in zip(all_preds, all_inputs, all_labels):
            pred, x, y = str(pred), str(x), str(y)
            if dataloader.dataset.is_correct(pred, y):
                num_correct += 1
            num_all += 1
            outputs.append((pred, x, y))

        acc = num_correct / num_all
        metrics['acc'] = acc
        print(f"Accuracy: {acc}")

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)