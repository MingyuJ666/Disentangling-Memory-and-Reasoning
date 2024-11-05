from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Optional
import transformers
import torch
import json
from torch.utils.data import DataLoader
import os, json, random, pickle,re
import numpy as np
from huggingface_hub import login
from load_data.preprocess import GSMData, AquaData, StrategyQAData, StrategyQAData_Ours, CommonsenseQAData_Ours, TruthfulQAData_Ours
from model.generation_utils import make_sparse_mask
from model.load_model import MyAutoModelForCausalLM
from model.peft_model import MyPeftModelForCausalLM

INVALID_ANS = "[invalid]"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "pre-trained language model name on Huggingface, or path to a checkpoint."},)
    base_model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "pre-trained language model name on Huggingface, or path to a checkpoint."},)
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default='./save_data', metadata={"help": "Path to the output dir."})
    save_result: Optional[bool] = field(default=True)
    max_length: Optional[int] = field(default=512)
    decoding_scheme: Optional[str] = field(default="greedy")
    load_in_8bit: Optional[bool] = field(default=False)
    use_calculator: Optional[bool] = field(default=False)
    add_soft_prompts: Optional[bool] = field(default=False)
    add_hard_prompts: Optional[bool] = field(default=False)
    only_at_front: Optional[bool] = field(default=False)
    use_sparse_attention: Optional[bool] = field(default=False)
    parameter_efficient_mode: Optional['str'] = field(default='none', 
        metadata={"choices": ["none", "prompt-tuning", "lora", "lora+prompt-tuning"]})
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Require for llama family."})
    enable_cpu_offload: Optional[bool] = field(default=False)
    only_at_front: Optional[bool] = field(default=False)
    plan_first: Optional[bool] = field(default=False)
    plan_only: Optional[bool] = field(default=False)
    extract_step_type_tokens: Optional[str] = field(default="none",
        metadata={"choices": ["none", "+-*/", "vae", "tf-idf", "k-means","memory","other"]})
    num_plan_types: Optional[int] = field(default=5)

@dataclass
class DataArguments:
    dataset: str = field(default=None, metadata={"help": "dataset name."})
    batch_size: Optional[int] = field(default=16)
    use_demonstrations: Optional[bool] = field(default=False)
    demo_selection: Optional[str] = field(default="uniform")
    candidate_size: Optional[int] = field(default=100)
    k_shot: Optional[int] = field(default=4)
    seed: Optional[int] = field(default=42)
    num_test: Optional[int] = field(default=1000)
    prompt_template: Optional[str] = field(default=None)
    embedding_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")


def main():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    login(token=model_args.hf_hub_token)
    print(model_args.extract_step_type_tokens)


    if model_args.output_dir is None:
        model_args.output_dir = model_args.model_name_or_path
    else:
        os.makedirs(model_args.output_dir, exist_ok = True)

    if 'llama2' in model_args.model_name_or_path or 'alpaca' in model_args.model_name_or_path:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    print("loaded tokenizer")

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    prompt_text = {}
    prompt_tokens = []
    num_new_tokens = 0
    step_type_predictor = None
    step_type_ids = None

    if model_args.add_soft_prompts:
        prompt_text_file =  f'{os.path.dirname(model_args.model_name_or_path)}/prompt_text.json'
        special_tokens_list = []

        if os.path.exists(prompt_text_file):

            prompt_text = json.load(open(prompt_text_file))
            for k in prompt_text:
                tokens = prompt_text[k].split('>')
                special_tokens_list += [tok+'>' for tok in tokens[:-1]]

            
            if "memory" in model_args.extract_step_type_tokens:
               
                class StepType:
                    def __init__(self):
                        self.vocab = ['reason','memory']

                    def predict(self, text: str, start=0):
                        
                        pattern = re.compile(r"^\[(.*?)\]:", re.MULTILINE)
                        matches = pattern.findall(text)

                        result = []
                        for match in matches:
                            if match == 'reason':
                                result.append('reason')
                            elif match == 'rag':
                                result.append('memory')
                        return result
                        
                
                step_type_predictor = StepType()

                for s in step_type_predictor.vocab:
                    prompt_text[s] = ''
                
        
        prompt_tokens = tokenizer.convert_tokens_to_ids(special_tokens_list)
        num_new_tokens = len(special_tokens_list)
    
    elif model_args.add_hard_prompts:
        if model_args.only_at_front and not model_args.plan_first:
            prompt_text = {'prefix': 'Plan: '}
        else:
            prompt_text = {'prefix': 'Plan: ', 'answer': ' answer', 'assignment': ' assignment', 
                                '+': ' addition ', '-': ' deduction', '*': ' multiplication', '/': ' division'}

    if model_args.parameter_efficient_mode != 'none':
        model_name = model_args.base_model_name_or_path
    else:
        model_name = model_args.model_name_or_path

    if 'prompt-tuning' in model_args.parameter_efficient_mode:
        input_embedding_file = model_args.model_name_or_path + '/embeddings.pt'
        output_embedding_file = None
        if not os.path.exists(input_embedding_file):
            input_embedding_file = model_args.model_name_or_path + '/input_embeddings.pt'
            output_embedding_file = model_args.model_name_or_path + '/output_embeddings.pt'
    else:
        input_embedding_file = None
        output_embedding_file = None

    if model_args.load_in_8bit:
        
        model = MyAutoModelForCausalLM.from_pretrained(n_tokens=num_new_tokens,
            input_embedding_file=input_embedding_file,
            output_embedding_file=output_embedding_file,
            sparse=model_args.use_sparse_attention,
            prompt_tokens=prompt_tokens,
            pretrained_model_name_or_path=model_name,
            parameter_efficient_mode=model_args.parameter_efficient_mode,
            cache_dir=model_args.cache_dir, torch_dtype=torch.float32, 
            device_map="auto", load_in_8bit=True,
            offload_folder="offload", offload_state_dict = True,
        )
       
    else:
        model = MyAutoModelForCausalLM.from_pretrained(n_tokens=num_new_tokens,
            input_embedding_file=input_embedding_file,
            output_embedding_file=output_embedding_file,
            sparse=model_args.use_sparse_attention,
            prompt_tokens=prompt_tokens,
            pretrained_model_name_or_path=model_name,
            parameter_efficient_mode=model_args.parameter_efficient_mode,
            cache_dir=model_args.cache_dir,
            device_map="auto", torch_dtype=torch.float32,  
            offload_folder="offload", offload_state_dict = True
        )
        
    
    if 'lora' in model_args.parameter_efficient_mode:
        model = MyPeftModelForCausalLM.from_pretrained(model, 
            model_args.model_name_or_path, 
            load_embeddings=model_args.add_soft_prompts, 
            n_tokens=num_new_tokens)
            
    print("loaded model.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()  
    
    if data_args.dataset == "gsm8k":
        data_class = GSMData
    elif data_args.dataset == "aqua":
        data_class = AquaData
    elif data_args.dataset == "qa":
        data_class = StrategyQAData
    elif data_args.dataset == "stratgeqa_agent":
        data_class = StrategyQAData_Ours
    elif data_args.dataset == "commonsenseqa_agent":
        data_class = CommonsenseQAData_Ours
    elif data_args.dataset == "truthfulqa_agent" or data_args.dataset == "truthfulqa_agent_crossdomain":
        data_class = TruthfulQAData_Ours

    dataset = data_class("test", prompt_text, 
                        add_soft_prompts=model_args.add_soft_prompts or model_args.add_hard_prompts, 
                        only_at_front=model_args.only_at_front,
                        plan_first=model_args.plan_first, 
                        plan_only=model_args.plan_only,
                        prompt_template=data_args.prompt_template,
                        step_type_ids=step_type_ids, tokenizer=tokenizer,
                        step_type_predictor=step_type_predictor,)
    random.seed(42)
    if len(dataset) > data_args.num_test:
        idx = random.choices(list(range(len(dataset))), k=data_args.num_test)
        new_x = []
        new_y = []
        for i in idx:
            new_x.append(dataset[i]['x'])
            new_y.append(dataset[i]['y'])
        dataset.x = new_x
        dataset.y = new_y
    assert len(dataset) <= data_args.num_test
    print(dataset[0], len(dataset))
    

    
    print("loaded dataset")
    
    dataloader = DataLoader(dataset, batch_size=data_args.batch_size, shuffle=False)


    prompt_ts = {}
    
    if step_type_predictor is not None:
        generated_planning_token_dist = {}
        gt_planning_token_dist ={}
        for k in step_type_predictor.vocab:
            prompt_ts[k] = prompt_text[k].strip().split('>')[0] + '>'
            

    num_correct = 0
    num_all = 0
    output_data = []
   


    for i, batch in tqdm(enumerate(dataloader)):
        x_text, y_text = batch['x'], batch['y']
        
        
        encoding = tokenizer(x_text, padding=True, return_tensors='pt').to(device)
        max_length = min(model_args.max_length, encoding['input_ids'].size(1) + 512)
        if model_args.use_sparse_attention:
            print("use sparse attention")
            sparese_attention_mask = make_sparse_mask(encoding['input_ids'], prompt_tokens).to(device)
            encoding["attention_mask"] = (encoding["attention_mask"], sparese_attention_mask)
        with torch.no_grad():
            generated_ids = model.generate(**encoding, 
                 max_length=model_args.max_length)


        try:
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print(generated_texts)
        except:
            print("cannot decode: ")
            print(generated_ids)

        for text, x, y in zip(generated_texts, x_text, y_text):
            text, x, y = str(text), str(x), str(y)
            
            if step_type_predictor is not None:
                for k in prompt_ts:
                    n_generated_k = text.count(prompt_ts[k])
                    n_gt_k = y.count(prompt_ts[k])
                    if k in generated_planning_token_dist:
                        generated_planning_token_dist[k] += n_generated_k
                    else:
                        generated_planning_token_dist[k] = n_generated_k
                    if k in gt_planning_token_dist:
                        gt_planning_token_dist[k] += n_gt_k
                    else:
                        gt_planning_token_dist[k] = n_gt_k
            print(text)
            result = ''
            if dataset.is_correct(text, y):
                num_correct += 1
                print('correct')
                result = 'correct'

            else:
                print('wrong')
                result = 'wrong'
            
            output_data.append({
            'generated_text': text,
            'result': result
            })
            
            num_all += 1


        print("Accuracy: ", num_correct/num_all)
        if step_type_predictor is not None:
            print("groundtruth planning token dist: ", gt_planning_token_dist)
            print("generated planning token dist: ", generated_planning_token_dist)
        
    
   
    if model_args.save_result:
        output_file = os.path.join(model_args.output_dir, f"{data_args.dataset}/{model_args.base_model_name_or_path}_output.json")
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print("Accuracy: ", num_correct/num_all)
    print("num test: ", num_all)
    



if __name__ == "__main__":
    main()