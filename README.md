# Disentangling Memory and Reasoning in LLM


## Set up 

All data and models can be loaded from Huggingface. A Hugging Face Hub token is required. 

To set up the environment, run
```
pip install -r requirements.txt 
```

Check the load_data file and get the dataset you want
```
cd load_data/

python data_agent.py 
```
You can check the readme.md in load_data file

## Training

The `train.sh` script can be used for (parameter efficient) fine-tuning a pre-trained language model with CoT data. The accuracy evaluation has been built in the training loop. If performing parameter efficient fine-tuning, only the adapters and the newly added token embeddings will be saved in the checkpoint in `./checkpoints`.

Key arguments:

* `--dataset`: Same as `--dataset` in `step_type.sh`.

* `--model_name_or_path`: Base model name/path.

* `--add_soft_prompts`: A binary argument determining whether to use planning tokens or not.

* `--parameter_efficient_mode`: Whether to perform parameter efficient fine-tuning. `none` means full fine-tuning. `lora` means LORA fine-tuning. `prompt-tuning` means tuning the embeddings of newly added tokens. To use planning tokens with LORA fine-tuning, this argument needs to be set as `lora+prompt-tuning`. The parameter efficient fine-tuning alghorithms are implemented based on the [PEFT](https://github.com/huggingface/peft) library. 

* `--num_general_prefix_tokens`: *n_prefix* in the paper.

* `--num_special_prefix_tokens`: *n_special* in the paper.

* `--extract_step_type_tokens`: same as `--selection_method` in `step_type.sh`.

* `--num_plan_types`: same as `--num_plan_types` in `step_type.sh`.

* `--model_max_length`: We set this to 512 for `gsm8k` and 1024 for `math` and `aqua`.

* `--num_test`: The maximum number of testing examples to be evaluated at the end of each epoch. We set this to 1000.

* `--int8_training`: Turn on this flag to save more GPU memories. Might impact the model performance.

## Evaluation

In the paper, we report the epoch producing the highest eval accuracy. The accuracy evaluation has been built in the training loop, and can be checked in the `trainer_state.json` file saved in the checkpoints as `eval_acc`.

To do evaluation sparately and save the generation results, use the script `eval.sh`. Need to make sure all the corresponding arguments are the same as the the training arguments in `train.sh`. The model generation results will be saved in the same directory as the loaded checkpoint. The saving format would be question - empty line - generated solution - empty line - ground truth solution.

Key arguments:

* `--model_name_or_path`: Checkpoint path. 

* `--base_model_name_or_path`: Base model for paramter efficient fine-tuning checkpoints. Must be the same as the `--model_name_or_path` in `train.sh` to correctly load the checkpoints.

* Please follow `train.sh` for other arguments.

## Where is the Checkpoint
