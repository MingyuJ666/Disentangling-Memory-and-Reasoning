DATASET=truthfulqa_agent
# MODEL=meta-llama/Llama-2-7b-chat-hf
MODEL=meta-llama/Llama-3.1-8B-Instruct
# MODEL=Qwen/Qwen2.5-7B-Instruct
ADD_SOFT_PROMPT=True
EFFICIENT=lora+prompt-tuning
STEP_TYPE=memory
CUDA_VISIBLE_DEVICES=2 python eval.py \
    --base_model_name_or_path $MODEL \
    --hf_hub_token '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' \
    --model_name_or_path /common/users/mj939/init_models/checkpoints/meta-llama/Llama-3.1-8B-Instruct/truthfulqa_agent/step_type=memory-3-4-efficient=lora+prompt-tuning-lr=2e-4-soft-prompt=True/checkpoint-585\
    --add_soft_prompts $ADD_SOFT_PROMPT\
    --parameter_efficient_mode $EFFICIENT \
    --dataset $DATASET \
    --batch_size 1 \
    --max_length 850 \
    --seed 300 \
    --extract_step_type_tokens $STEP_TYPE \
    --embedding_model_name $MODEL \
    --num_plan_types 5 \
    --num_test 1200 \
    --load_in_8bit True \
    # --prompt_template alpaca \
    # --use_calculator True \
    #几个很好的checkpoint Qwen-801（/common/users/mj939/init_models/checkpoints/Qwen/Qwen2.5-7B-Instruct/stratgeqa_agent/step_type=memory-3-4-efficient=lora+prompt-tuning-lr=2e-4-soft-prompt=True）
    # /common/users/mj939/init_models/checkpoints/meta-llama/Llama-3.1-8B-Instruct/commonsenseqa_agent/step_type=memory-3-4-efficient=lora+prompt-tuning-lr=2e-4-soft-prompt=True/checkpoint-1466\
    
    
    
    # truthfulqa_agent
    # /common/users/mj939/init_models/checkpoints/meta-llama/Llama-2-7b-chat-hf/truthfulqa_agent/step_type=memory-3-4-efficient=lora+prompt-tuning-lr=2e-4-soft-prompt=True/checkpoint-395
    # /common/users/mj939/init_models/checkpoints/Qwen/Qwen2.5-7B-Instruct/truthfulqa_agent/step_type=memory-3-4-efficient=lora+prompt-tuning-lr=2e-4-soft-prompt=True/checkpoint-585\
   