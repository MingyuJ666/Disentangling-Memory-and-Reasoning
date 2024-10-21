DATASET=stratgeqa_agent
MODE=supervised
MODEL=meta-llama/Llama-2-7b-chat-hf
ADD_SOFT_PROMPT=True
N_PREFIX=2
N_SPECIAL=3
EFFICIENT=lora+prompt-tuning
STEP_TYPE=memory
LR=2e-4
CUDA_VISIBLE_DEVICES=3 python train.py \
    --model_name_or_path $MODEL \
    --add_soft_prompts $ADD_SOFT_PROMPT\
    --hf_hub_token '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'\
    --num_general_prefix_tokens $N_PREFIX \
    --num_special_prefix_tokens $N_SPECIAL \
    --parameter_efficient_mode $EFFICIENT \
    --dataset $DATASET \
    --load_in_16fp False \
    --output_dir ./checkpoints/$MODEL/$DATASET/step_type=$STEP_TYPE-$N_PREFIX-$N_SPECIAL-efficient=$EFFICIENT-lr=$LR-soft-prompt=$ADD_SOFT_PROMPT\
    --model_max_length 700 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 200 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_steps 500 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --optim "adamw_torch" \
    --gradient_accumulation_steps 16 \
    --embedding_model_name $MODEL \
    --extract_step_type_tokens $STEP_TYPE \
    --num_plan_types 5 \
    --num_test 100 \
    --lora_module mlp \
    --int8_training True \
    # --gradient_checkpointing \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --sharded_ddp "zero_dp_2 offload" \
    # --fsdp "full_shard offload" \