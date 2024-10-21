DATASET=stratgeqa_agent
MODEL=meta-llama/Llama-2-7b-chat-hf
ADD_SOFT_PROMPT=True
N_PREFIX=2
N_SPECIAL=3
EFFICIENT=lora+prompt-tuning
STEP_TYPE=memory
CUDA_VISIBLE_DEVICES=1 python eval.py \
    --base_model_name_or_path $MODEL \
    --hf_hub_token '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' \
    --model_name_or_path  /common/home/mj939/planning_tokens/checkpoints/meta-llama/Llama-2-7b-chat-hf/stratgeqa_agent/step_type=memory-2-3-efficient=lora+prompt-tuning-lr=2e-4-soft-prompt=True/checkpoint-1000 \
    --add_soft_prompts $ADD_SOFT_PROMPT\
    --num_general_prefix_tokens $N_PREFIX \
    --num_special_prefix_tokens $N_SPECIAL \
    --parameter_efficient_mode $EFFICIENT \
    --dataset $DATASET \
    --batch_size 1 \
    --max_length 700 \
    --seed 300 \
    --extract_step_type_tokens $STEP_TYPE \
    --embedding_model_name $MODEL \
    --num_plan_types 5 \
    --num_test 1000 \
    --load_in_8bit True \
    # --prompt_template alpaca \
    # --use_calculator True \