import random, copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
import transformers
import torch

from model.generation_utils import make_sparse_mask

IGNORE_INDEX = -100

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]       
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def prepare_data(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) 
                                             for strings in (examples, sources)]
    eos = torch.tensor([tokenizer.eos_token_id])
    input_ids = [torch.cat((ids, eos)) for ids in examples_tokenized["input_ids"]]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        self.data = dataset
        data_dict = prepare_data(self.data.x, self.data.y, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    prompt_tokens: Sequence[int]
    use_sparse_attention: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] 
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, 
                                                 padding_value=IGNORE_INDEX)

        if self.use_sparse_attention:
            sparse_masks = make_sparse_mask(input_ids, self.prompt_tokens)
            attn_mask = (input_ids.ne(self.tokenizer.pad_token_id), sparse_masks)
        else:
            attn_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attn_mask,
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, 
                                dataset, eval_dataset, prompt_tokens, use_sparse_attention,
                                max_num_eval=1000) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    random.seed(42)
    if eval_dataset is not None and len(eval_dataset) > max_num_eval:
        idx = random.choices(list(range(len(eval_dataset))), k=max_num_eval)
        new_x = []
        new_y = []
        for i in idx:
            new_x.append(eval_dataset[i]['x'])
            new_y.append(eval_dataset[i]['y'])
        eval_dataset.x = new_x
        eval_dataset.y = new_y
    assert len(eval_dataset) <= max_num_eval
    train_dataset = SupervisedDataset(dataset, tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, 
                    prompt_tokens=prompt_tokens, use_sparse_attention=use_sparse_attention)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, 
                data_collator=data_collator)