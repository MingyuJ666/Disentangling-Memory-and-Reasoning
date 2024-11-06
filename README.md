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

### For Train LLAMA2-7B in the StrategyQA dataset: you can follow this setting to get the reported score:
DATASET=stratgeqa_agent\
MODE=supervised\
MODEL=meta-llama/Llama-2-7b-chat-hf\
ADD_SOFT_PROMPT=True\
N_PREFIX=2\
N_SPECIAL=4\
EFFICIENT=lora+prompt-tuning\
STEP_TYPE=memory\
LR=2e-4\
CUDA_VISIBLE_DEVICES=1 python train.py \\\
    --model_name_or_path $MODEL \\\
    --add_soft_prompts $ADD_SOFT_PROMPT \\\
    --hf_hub_token 'your token'\\\
    --num_general_prefix_tokens $N_PREFIX \\\
    --num_special_prefix_tokens $N_SPECIAL \\\
    --parameter_efficient_mode $EFFICIENT \\\
    --dataset $DATASET \\\
    --load_in_16fp False \\\
    --output_dir ./checkpoints/$MODEL/$DATASET/step_type=$STEP_TYPE-$N_PREFIX-$N_SPECIAL-efficient=$EFFICIENT-lr=$LR-soft-prompt=$ADD_SOFT_PROMPT \\\
    --model_max_length 700 \\\
    --num_train_epochs 10 \\\
    --per_device_train_batch_size 1 \\\
    --per_device_eval_batch_size 1 \\\
    --evaluation_strategy "epoch" \\\
    --save_strategy "epoch" \\\
    --save_total_limit 200 \\\
    --learning_rate $LR \\\
    --weight_decay 0. \\\
    --warmup_steps 700 \\\
    --lr_scheduler_type "cosine" \\\
    --logging_steps 1 \\\
    --optim "adamw_torch" \\\
    --gradient_accumulation_steps 16 \\\
    --embedding_model_name $MODEL \\\
    --extract_step_type_tokens $STEP_TYPE \\\
    --num_plan_types 5 \\\
    --num_test 50 \\\
    --lora_module mlp \\\
    --int8_training True \

## Evaluation

In the paper, we report the epoch producing the highest eval accuracy. The accuracy evaluation has been built in the training loop, and can be checked in the `trainer_state.json` file saved in the checkpoints as `eval_acc`.

To do evaluation sparately and save the generation results, use the script `eval.sh`. Need to make sure all the corresponding arguments are the same as the the training arguments in `train.sh`. The model generation results will be saved in the same directory as the loaded checkpoint. The saving format would be question - empty line - generated solution - empty line - ground truth solution.

Key arguments:

* `--model_name_or_path`: Checkpoint path. 

* `--base_model_name_or_path`: Base model for paramter efficient fine-tuning checkpoints. Must be the same as the `--model_name_or_path` in `train.sh` to correctly load the checkpoints.

* Please follow `train.sh` for other arguments.

### To evaluate LLAMA2-7B in the StrategyQA dataset: you can follow this setting to get the reported score(0.706):
DATASET=stratgeqa_agent\
MODEL=meta-llama/Llama-2-7b-chat-hf\
ADD_SOFT_PROMPT=True\
EFFICIENT=lora+prompt-tuning\
STEP_TYPE=memory\
CUDA_VISIBLE_DEVICES=0 python eval.py \\\
    --base_model_name_or_path $MODEL \\\
    --hf_hub_token ....... \\\
    --model_name_or_path ./checkpoints/meta-llama/Llama-2-7b-chat-hf/stratgeqa_agent/step_type=memory-2-4-efficient=lora+prompt-tuning-lr=2e-4-soft-prompt=True/checkpoint-801\\\
    --add_soft_prompts $ADD_SOFT_PROMPT\\\
    --parameter_efficient_mode $EFFICIENT \\\
    --dataset $DATASET \\\
    --batch_size 1 \\\
    --max_length 700 \\\
    --seed 300 \\\
    --extract_step_type_tokens $STEP_TYPE \\\
    --embedding_model_name $MODEL \\\
    --num_plan_types 5 \\\
    --num_test 1000 \\\
    --load_in_8bit True \

## Where is the Checkpoint
you can find Llama2-7b's checkpoint on StrategyQA here: https://drive.google.com/drive/folders/1w1Ukd7LOlqlUue8uTf8hjEiz12S2Zwnw?usp=drive_link, checkpoint-801 is the best one

you can find Llama3.1-8b's checkpoint on CommenseQA here: https://drive.google.com/drive/folders/1w1Ukd7LOlqlUue8uTf8hjEiz12S2Zwnw?usp=drive_link, checkpoint-801 is the best one
