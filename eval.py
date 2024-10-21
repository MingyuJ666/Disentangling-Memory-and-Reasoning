from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Optional
import transformers
import sentence_transformers
import torch
from torch.utils.data import DataLoader
import os, json, random, pickle
import numpy as np
from huggingface_hub import login

from load_data.preprocess import GSMData, AquaData, StrategyQAData, StrategyQAData_Ours
from load_data.k_shot_dataset import KshotDataset
import calculator
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
    output_dir: Optional[str] = field(default=None, metadata={"help": "Path to the output dir."})
    max_length: Optional[int] = field(default=512)
    decoding_scheme: Optional[str] = field(default="greedy")
    load_in_8bit: Optional[bool] = field(default=False)
    use_calculator: Optional[bool] = field(default=False)
    num_general_prefix_tokens: Optional[int] = field(default=5)
    num_special_prefix_tokens: Optional[int] = field(default=1)
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
        metadata={"choices": ["none", "+-*/", "vae", "tf-idf", "k-means","memory"]})
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

    if model_args.output_dir is None:
        model_args.output_dir = model_args.model_name_or_path
    else:
        os.makedirs(model_args.output_dir, exist_ok = True)

    if 'llama' in model_args.model_name_or_path or 'alpaca' in model_args.model_name_or_path:
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

        else:
            if "+-*/" in model_args.extract_step_type_tokens:
                if data_args.dataset == "gsm8k":
                    prompt_text = {'prefix': '', 'answer': '', 'assignment': '', 
                                    '+': '', '-': '', '*': '', '/': ''}
                elif data_args.dataset in ["math", "aqua"]:
                    prompt_text = {'prefix': '', 'answer': '', 'assignment': '', 
                                    '+': '', '-': '', '*': '', '/': '',
                                    '^': '', '!': '',
                                    'sqrt': '', 'frac': '', 'cos': '', 'sin': '',
                                    'tan': '', 'log': '', 'ln': '', 'times': ''}
                else:
                    raise NotImplementedError 
            else:
                prompt_text = {'prefix': '', 'answer': '', 'assignment': ''}

            
            if "k-means" in model_args.extract_step_type_tokens or 'vae' in model_args.extract_step_type_tokens:
                with open("/common/home/mj939/planning_tokens/load_data/extract_steps/meta-llama/Llama-2-7b-chat-hf/qa/vae_5/epoch9/qa_vae_5.pkl", 'rb') as f:
                    cluster_model = pickle.load(f)
                
                class StepType:
                    def __init__(self, cluster_model):
                        self.cluster_model = cluster_model
                        self.vocab = cluster_model.get_feature_names_out()
                        self.tokenizer = None
                        if data_args.embedding_model_name == 'all-mpnet-base-v2':
                            self.embedding_model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
                        elif 't5' in data_args.embedding_model_name:
                            self.embedding_model = transformers.T5EncoderModel.from_pretrained(data_args.embedding_model_name).to('cuda')
                            self.tokenizer = transformers.AutoTokenizer.from_pretrained(data_args.embedding_model_name, legacy=False)
                        elif 'llama' in data_args.embedding_model_name or 'gpt2' in data_args.embedding_model_name:
                            self.embedding_model = transformers.AutoModelForCausalLM.from_pretrained(data_args.embedding_model_name).to('cuda')
                            self.tokenizer = transformers.AutoTokenizer.from_pretrained(data_args.embedding_model_name, legacy=False)

                        if self.tokenizer is not None and self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token_id = 0
                        self.embedding_model.eval()

                    def predict(self, text: str, start=0):
                        with torch.no_grad():

                            if data_args.embedding_model_name == 'all-mpnet-base-v2':
                                embedding = self.embedding_model.encode(text.split('\n')[start:-1])

                            elif 't5' in data_args.embedding_model_name:
                                inputs = self.tokenizer(text.split('\n')[start:-1], return_tensors="pt", 
                                    padding="longest", max_length=model_args.max_length, 
                                    truncation=True,).to('cuda')
                                outputs = self.embedding_model(**inputs)
                                last_hidden_states = outputs.last_hidden_state
                                mean_hidden = torch.sum(last_hidden_states 
                                    * inputs["attention_mask"].unsqueeze(-1), 1)\
                                        /torch.sum(inputs["attention_mask"], -1).unsqueeze(-1)
                                embedding = mean_hidden.cpu().numpy()

                            elif 'llama' in data_args.embedding_model_name or 'gpt2' in data_args.embedding_model_name:
                                inputs = self.tokenizer([text], return_tensors="pt", 
                                    padding="longest", max_length=model_args.max_length, 
                                    truncation=True,).to('cuda')
                                outputs = self.embedding_model(**inputs, 
                                    output_hidden_states=True, return_dict=True)
                                last_hidden_state = outputs.hidden_states[-1][0]
                                if 'llama' in data_args.embedding_model_name:
                                    split_id = 13
                                elif 'gpt2' in data_args.embedding_model_name:
                                    split_id = 198
                                step_mask = torch.cumsum(inputs['input_ids'][0]==split_id, dim=-1)
                                embedding = []
                                for j in range(start, torch.max(step_mask)):
                                    step_j_mask = (step_mask == j).int()
                                    step_j_rep = (last_hidden_state * step_j_mask.unsqueeze(-1)).sum(0)
                                    step_len = step_j_mask.sum()
                                    if step_len > 0:
                                        embedding.append((step_j_rep/step_len).cpu().numpy())
                                    else:
                                        print("current step is empty")
                                embedding = np.array(embedding)
                                
                            if 'vae' in model_args.extract_step_type_tokens:
                                embedding = torch.from_numpy(embedding).to('cuda')
                                label = self.cluster_model.predict(embedding).cpu().numpy()
                            else:
                                label = self.cluster_model.predict(embedding)

                        return [self.vocab[l] for l in label]
                
                step_type_predictor = StepType(cluster_model)

                for s in step_type_predictor.vocab:
                    prompt_text[s] = ''

            elif "memory" in model_args.extract_step_type_tokens:
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


            for k in prompt_text:
                text = ''
                if k == 'prefix':
                    num_tokens = model_args.num_general_prefix_tokens
                else:
                    num_tokens = model_args.num_special_prefix_tokens
                for i in range(num_tokens):
                    token_name = f'<{k}_{i}>'
                    special_tokens_list.append(token_name)
                    text += token_name
                prompt_text[k] = text
        
        prompt_tokens = tokenizer.convert_tokens_to_ids(special_tokens_list)
        print(prompt_text)
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
        quantization_config = transformers.BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=model_args.enable_cpu_offload)
        model = MyAutoModelForCausalLM.from_pretrained(n_tokens=num_new_tokens,
            input_embedding_file=input_embedding_file,
            output_embedding_file=output_embedding_file,
            sparse=model_args.use_sparse_attention,
            prompt_tokens=prompt_tokens,
            pretrained_model_name_or_path=model_name,
            parameter_efficient_mode=model_args.parameter_efficient_mode,
            cache_dir=model_args.cache_dir,
            device_map="auto", load_in_8bit=True,
            offload_folder="offload", offload_state_dict = True,
            quantization_config=quantization_config
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

    if data_args.use_demonstrations:
        out_file_name = f'{model_args.output_dir}/{data_args.dataset}_test_cal={model_args.use_calculator}_demo={data_args.demo_selection}_k={data_args.k_shot}_output.txt'
    elif model_args.add_soft_prompts:
        out_file_name = f'{model_args.output_dir}/{data_args.dataset}_test_cal={model_args.use_calculator}_soft_prompt_output.txt'
    else:
        out_file_name = f'{model_args.output_dir}/{data_args.dataset}_test_cal={model_args.use_calculator}_output.txt'

    prompt_ts = {}
    if step_type_predictor is not None:
        generated_planning_token_dist = {}
        gt_planning_token_dist ={}
        for k in step_type_predictor.vocab:
            prompt_ts[k] = prompt_text[k].strip().split('>')[0] + '>'
            
    output = []
    num_correct = 0
    num_all = 0

    for i, batch in tqdm(enumerate(dataloader)):
        x_text, y_text = batch['x'], batch['y']
        print(x_text)
        print("##############")
        print(y_text)
        
        
        encoding = tokenizer(x_text, padding=True, return_tensors='pt').to(device)
        max_length = min(model_args.max_length, encoding['input_ids'].size(1) + 512)
        if model_args.use_sparse_attention:
            print("use sparse attention")
            sparese_attention_mask = make_sparse_mask(encoding['input_ids'], prompt_tokens).to(device)
            encoding["attention_mask"] = (encoding["attention_mask"], sparese_attention_mask)
        with torch.no_grad():
            generated_ids = model.generate(**encoding, 
                max_length=model_args.max_length)

        # generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
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
            if dataset.is_correct(text, y):
                num_correct += 1
                print('correct')
            else:
                print('wrong')
            
            num_all += 1
            output.append((text, y))

        with open(out_file_name, 'w') as f:
            for x, y in output:
                f.write(x.encode('ascii', 'ignore').decode('ascii') + '\n' + 
                        y.encode('ascii', 'ignore').decode('ascii') + '\n\n')
            f.write(f"Accuracy: {num_correct/num_all}")
        
        print("Accuracy: ", num_correct/num_all)
        if step_type_predictor is not None:
            print("groundtruth planning token dist: ", gt_planning_token_dist)
            print("generated planning token dist: ", generated_planning_token_dist)
    
    print("Accuracy: ", num_correct/num_all)
    print("num test: ", num_all)
    
    with open(out_file_name, 'w') as f:
        for x, y in output:
            f.write(x.encode('ascii', 'ignore').decode('ascii') + '\n' + 
                    y.encode('ascii', 'ignore').decode('ascii') + '\n\n')
        f.write(f"Accuracy: {num_correct/num_all}")
        if step_type_predictor is not None:
            f.write(f"groundtruth planning token dist: {gt_planning_token_dist}")
            f.write(f"generated planning token dist: {generated_planning_token_dist}")


if __name__ == "__main__":
    main()