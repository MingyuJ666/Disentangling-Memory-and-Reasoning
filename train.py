from dataclasses import dataclass, field
from typing import Optional
import torch
import re
import json, pickle
import numpy as np
import torch
import transformers
import sentence_transformers
from model.my_trainer import MyTrainer
transformers.logging.set_verbosity_info()
from huggingface_hub import login
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model, PrefixTuningConfig, PromptTuningConfig, PromptTuningInit
from transformers.utils import logging
from model.load_model import MyAutoModelForCausalLM
from load_data.supervised_dataset import make_supervised_data_module
from load_data.constant_len_dataset import make_constant_len_data_module
from load_data.preprocess import GSMData, AquaData, StrategyQAData, StrategyQAData_Ours
from model.peft_model import MyPeftModelForCausalLM
from huggingface_hub import login



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device: ", device)
print(torch.cuda.is_available())


INVALID_ANS = "[invalid]"

logger = logging.get_logger(__name__)

@dataclass
class ModelArguments:
    random_initialize: Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "pre-trained language model name on Huggingface, or path to a checkpoint."},)
    hf_hub_token: Optional[str] = field(default='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')
    num_general_prefix_tokens: Optional[int] = field(default=3)
    num_special_prefix_tokens: Optional[int] = field(default=3)
    add_soft_prompts: Optional[bool] = field(default=False)
    add_hard_prompts: Optional[bool] = field(default=False)
    use_sparse_attention: Optional[bool] = field(default=False)
    parameter_efficient_mode: Optional['str'] = field(default='none', 
        metadata={"choices": ["none", "prompt-tuning", "lora", "lora+prompt-tuning", "prefix-tuning"]})
    only_at_front: Optional[bool] = field(default=False)
    plan_first: Optional[bool] = field(default=False)
    plan_only: Optional[bool] = field(default=False)
    use_calculator: Optional[bool] = field(default=False)
    decoding_scheme: Optional[str] = field(default="greedy")
    extract_step_type_tokens: Optional[str] = field(default="none",
        metadata={"choices": ["none", "+-*/", "vae", "tf-idf", "k-means","mlp",'memory']})
    num_plan_types: Optional[int] = field(default=5)
    lora_module: Optional[str] = field(default="mlp")

@dataclass
class DataArguments:
    dataset: str = field(default="gsm8k", 
        metadata={"help": "dataset name on Huggingface.", "choices": ["gsm8k", "aqua", "math","qa","stratgeqa_agent","qabig"]})
    mode: str = field(default="supervised", metadata={"choices": ["supervised", "constant_len"]})
    use_demonstrations: Optional[bool] = field(default=False)
    demo_selection: Optional[str] = field(default="uniform")
    candidate_size: Optional[int] = field(default=100)
    k_shot: Optional[int] = field(default=4)
    num_test: Optional[int] = field(default=1000)
    prompt_template: Optional[str] = field(default=None)
    embedding_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512,
        metadata={"help": "Maximum sequence length. Sequences will be left padded (and possibly truncated)."},
    )
    resume: Optional[bool] = field(default=False, metadata={"help": "Resume training from a checkpoint."})
    int8_training: Optional[bool] = field(default=False)
    load_in_16fp: Optional[bool] = field(default=False)
    load_in_32fp: Optional[bool] = field(default=False)
    num_train_epochs: Optional[int] = field(default=1)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def enable_prompt_tuning(model):
    model.get_input_embeddings().new_embedding.weight.requires_grad = True
    model.get_output_embeddings().new_linear.weight.requires_grad = True
    

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    login(token=model_args.hf_hub_token)


    if 'llama' in model_args.model_name_or_path or 'alpaca' in model_args.model_name_or_path:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    
    

    tokenizer.model_max_length = training_args.model_max_length
    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = '</s>'
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    prompt_text = {}
    prompt_tokens = []
    initialize_words_list = []
    initialize_tokens = None
    step_type_ids = None
    step_type_predictor = None

    if model_args.add_soft_prompts:
        
        if (model_args.only_at_front and not model_args.plan_first) or model_args.num_special_prefix_tokens==0:
            prompt_text = {'prefix': ''}
            
        else:
            prompt_text = {'prefix': '', 'answer': '', 'assignment': ''}
          

            if "memory" in model_args.extract_step_type_tokens:
                
                class StepType:
                    def __init__(self):
                        self.vocab = ['reason','rag']

                    def predict(self, text: str, start=0):
                        
                        pattern = re.compile(r"^\[(.*?)\]:", re.MULTILINE)
                        matches = pattern.findall(text)

                        result = []
                        for match in matches:
                            if match == 'reason':
                                result.append('reason')
                            elif match == 'rag':
                                result.append('rag')
                        return result
                        
                
                step_type_predictor = StepType()

                for s in step_type_predictor.vocab:
                    prompt_text[s] = ''

        
        special_tokens_list = []
        
        for k in prompt_text:
            text = ''
            if k == 'prefix':
                num_tokens = model_args.num_general_prefix_tokens
            else:
                num_tokens = model_args.num_special_prefix_tokens
            for i in range(num_tokens):
                token_name = f'<{k}_{i}>'
                special_tokens_list.append(token_name)
                initialize_words_list.append(k)
                text += ' ' + token_name
            prompt_text[k] = text
        
        

        num_new_tokens += tokenizer.add_tokens(special_tokens_list)
        prompt_tokens = tokenizer.convert_tokens_to_ids(special_tokens_list)
        initialize_tokens = tokenizer.convert_tokens_to_ids(initialize_words_list)
        assert len(prompt_tokens) == len(initialize_tokens)
          

    elif model_args.add_hard_prompts:
        
        if model_args.only_at_front and not model_args.plan_first:
            prompt_text = {'prefix': 'Plan:'}
        else:
            prompt_text = {'prefix': 'Plan:', 'answer': ' answer', 'assignment': ' assignment', '+': ' addition ', '-': ' deduction', '*': ' multiplication', '/': ' division'}
            
    
     
    if data_args.dataset == "gsm8k":
        data_class = GSMData
    elif data_args.dataset == "aqua":
        data_class = AquaData
    elif data_args.dataset == "qa":
        data_class = StrategyQAData
    elif data_args.dataset == "qabig":
        data_class = StrategyQAData    
    elif data_args.dataset == "stratgeqa_agent":
        data_class = StrategyQAData_Ours

    else:
        raise NotImplementedError
    

    if "agent" in data_args.dataset:
        
        dataset = data_class("train", prompt_text, 
                            add_soft_prompts=model_args.add_soft_prompts or model_args.add_hard_prompts, 
                            only_at_front=model_args.only_at_front,
                            plan_first=model_args.plan_first, 
                            plan_only=model_args.plan_only,
                            prompt_template=data_args.prompt_template,
                            step_type_ids=step_type_ids, tokenizer=tokenizer,
                            step_type_predictor=step_type_predictor)
       
        
        
    else:
        dataset = data_class("train", prompt_text, 
                            add_soft_prompts=model_args.add_soft_prompts or model_args.add_hard_prompts, 
                            only_at_front=model_args.only_at_front,
                            plan_first=model_args.plan_first, 
                            plan_only=model_args.plan_only,
                            prompt_template=data_args.prompt_template,
                            step_type_ids=step_type_ids, tokenizer=tokenizer,
                            step_type_predictor=step_type_predictor)
        
   

        
    
    eval_dataset = data_class("test", prompt_text, 
                        add_soft_prompts=model_args.add_soft_prompts or model_args.add_hard_prompts, 
                        only_at_front=model_args.only_at_front,
                        plan_first=model_args.plan_first, 
                        plan_only=model_args.plan_only,
                        prompt_template=data_args.prompt_template,
                        step_type_ids=step_type_ids, tokenizer=tokenizer,
                        step_type_predictor=step_type_predictor,)
    
    
  
    if "agent" in data_args.dataset:
        step_type_predictor = None
    else:
        if step_type_predictor is not None:
            del step_type_predictor.embedding_model
            del step_type_predictor.cluster_model
            step_type_predictor = None
    

    if data_args.mode == "supervised":
        data_module = make_supervised_data_module(tokenizer, dataset, eval_dataset, 
                    prompt_tokens, model_args.use_sparse_attention, max_num_eval=data_args.num_test)
       
    elif data_args.mode == "constant_len":
        data_module = make_constant_len_data_module(tokenizer, dataset, eval_dataset,
                                                    training_args.model_max_length)
    

    if training_args.load_in_16fp or training_args.int8_training:
        print("loading in 16fp",training_args.load_in_16fp)
        print("int8 training",training_args.int8_training)

        model = MyAutoModelForCausalLM.from_pretrained(n_tokens=num_new_tokens,
            initialize_tokens=initialize_tokens,
            sparse=model_args.use_sparse_attention,
            parameter_efficient_mode=model_args.parameter_efficient_mode,
            pretrained_model_name_or_path=model_args.model_name_or_path,
            cache_dir=training_args.cache_dir, torch_dtype=torch.float16, 
            device_map="auto", load_in_8bit=training_args.int8_training,
            offload_folder="offload", offload_state_dict = True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        
        
        
    else:
        model = MyAutoModelForCausalLM.from_pretrained(n_tokens=num_new_tokens,
            initialize_tokens=initialize_tokens,
            sparse=model_args.use_sparse_attention,
            parameter_efficient_mode=model_args.parameter_efficient_mode,
            pretrained_model_name_or_path=model_args.model_name_or_path,
            cache_dir=training_args.cache_dir, 
        )
   
    if 'lora' in model_args.parameter_efficient_mode:
        if 'llama' in model_args.model_name_or_path or 'alpaca' in model_args.model_name_or_path:
            target_modules = []
            if model_args.lora_module == 'mlp':
                target_modules += ["gate_proj", "up_proj", "down_proj"]
            if model_args.lora_module == 'atten':
                target_modules += ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif 'gpt2' in model_args.model_name_or_path:
            target_modules = ["c_attn", "c_proj"]
        else:
            raise NotImplementedError
        
        peft_config = LoraConfig(r=16, lora_alpha=16, target_modules=target_modules, 
                                lora_dropout=0.05, bias="none", inference_mode=False,
                                task_type=TaskType.CAUSAL_LM)
        model = MyPeftModelForCausalLM(model, peft_config, add_tokens=model_args.add_soft_prompts)
        if "prompt-tuning" in model_args.parameter_efficient_mode:
            enable_prompt_tuning(model.base_model.model)


    elif 'prompt-tuning' in model_args.parameter_efficient_mode:
        if model_args.only_at_front:
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=model_args.num_general_prefix_tokens,
                prompt_tuning_init_text="Solve the following math problem step-by-step:",
                tokenizer_name_or_path=model_args.model_name_or_path,
            )
            model = get_peft_model(model, peft_config)
        else:
            for p in model.parameters():
                p.requires_grad = False
            enable_prompt_tuning(model)
    
   


    trainer = MyTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
   
    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    if model_args.add_soft_prompts or model_args.add_hard_prompts:
        with open(f'{training_args.output_dir}/prompt_text.json', 'w') as f:
            json.dump(prompt_text, f, indent=4)


if __name__ == "__main__":
    train()
