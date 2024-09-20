import re, os, json
from datasets import load_dataset
from torch.utils.data import Dataset
from utils import _strip_string, delete_extra_zero, compare_both_string_and_number_format

class BaseData(Dataset):

    def __init__(self, split: str, soft_prompt_text: list, 
                 invalid_ans="[invalid]", add_soft_prompts=False, 
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):
        super(BaseData, self).__init__()
        self.data = self.load_data(split)
        self.split = split
        
        self.soft_prompt_text = soft_prompt_text
        self.INVALID_ANS = invalid_ans
        self.ANS_RE = re.compile(r"The answer is: (\-?[0-9\.\,]+)")
        self.add_soft_prompts = add_soft_prompts
        self.only_at_front = only_at_front
        self.plan_first = plan_first
        self.plan_only = plan_only
        self.step_type_ids = step_type_ids
        self.tokenizer = tokenizer
        self.step_type_predictor = step_type_predictor
        self.step_type_re = None
        
        if self.add_soft_prompts:
            if step_type_ids is None:
                skip_list = ['prefix', 'answer', 'assignment']
                if self.step_type_predictor is not None:
                    skip_list += list(self.step_type_predictor.vocab)
                if len(soft_prompt_text) > len(skip_list):
                    self.step_type_re = r"{}".format('|'.join([re.escape(x) 
                        for x in soft_prompt_text if x not in skip_list]))
            elif self.step_type_predictor is None:
                assert tokenizer is not None, "Tokenizer must be provided if step_type_ids is not None."
        print(self.step_type_re)

        if prompt_template is not None:
            file_name = f"./load_data/prompt_templates/{prompt_template}.json"
            assert os.path.exists(file_name), f"Prompt template {prompt_template} not found."
            self.prompt_template = json.load(open(file_name, "r"))
        else:
            self.prompt_template = None

        self.x = []
        self.y = []
        self.prepare_data()
        assert len(self.x) == len(self.y)

    def load_data(self, split: str):
        raise NotImplementedError
    
    def parse_q_a(self, example):
        raise NotImplementedError

    def extract_step_type(self, q, cot_steps):
        if len(cot_steps) == 0:
            return None
        
        step_types = [['prefix'] for _ in cot_steps]
        if self.step_type_predictor is not None:
            text = "Question: " + q + '\n ' + ' \n'.join(cot_steps) + '\n'
            start = len(q.split('\n'))
            try:
                new_step_types = self.step_type_predictor.predict(text, start)
            except:
                return None
            if len(new_step_types) != len(cot_steps):
                print(new_step_types)
                print(cot_steps)
                if len(new_step_types) >  len(cot_steps):
                    exit(1)
            for i, t in enumerate(new_step_types):
                step_types[i].append(t)

        if self.step_type_ids is None:
            if self.step_type_re is not None:
                for i, step in enumerate(cot_steps):
                    step_types[i] += re.findall(self.step_type_re, step) 
        else:
            for i, step in enumerate(cot_steps):
                text_ids = self.tokenizer.encode(step)
                for j in self.step_type_ids:
                    if j in text_ids:
                        step_types[i].append(self.tokenizer.convert_ids_to_tokens(i))

        for i, step in enumerate(cot_steps):
            if len(step_types[i]) == 1:
                step_types[i].append('assignment')

        return step_types
    
    def prepare_data(self):
        i = 0
        for ex_idx, ex in enumerate(self.data):
            print(f'{ex_idx} / {len(self.data)}', end='\r')
            plans = ''
            sol = ''
            inst, q, cot_steps, ans = self.parse_q_a(ex)
            if cot_steps is None:
                continue

            if self.add_soft_prompts:
                if self.only_at_front:
                    sol += self.soft_prompt_text['prefix'] + ' ' + ' \n'.join(cot_steps) + '\n'
                    i += len(cot_steps)
                else:
                    prompt_types = self.extract_step_type(q, cot_steps)
                    if prompt_types is None:
                        continue
                    # print(prompt_types)
                    # print(cot_steps)
                    for step, prompt_type in zip(cot_steps, prompt_types):
                        if len(self.soft_prompt_text) > 1:
                            for p_type in prompt_type:
                                sol += self.soft_prompt_text[p_type]
                                plans += self.soft_prompt_text[p_type]
                            plans += '\n'
                        else:
                            sol += self.soft_prompt_text['prefix']
                        sol += ' ' + step + '\n'
                        i += 1
            
            else:
                sol += ' \n'.join(cot_steps) + '\n'
                i += len(cot_steps)

            if self.add_soft_prompts:
                if not self.only_at_front:
                    sol += self.soft_prompt_text['prefix']
                    if len(self.soft_prompt_text) > 1:
                        sol += self.soft_prompt_text['answer']
                        plans += self.soft_prompt_text['answer'] + '\n'
                sol += ' The answer is: ' + ans + '\n'
                if self.plan_first:
                    sol = plans + sol
                if self.plan_only:
                    sol = plans + ' The answer is: ' + ans + '\n '
            else:
                sol += ' The answer is: ' + ans + '\n '
            i += 1

            if i < 1000:
                print(sol)
            self.y.append(sol)

            if self.prompt_template is not None:
                x = self.prompt_template["prompt_input"].format(
                    instruction=inst, input=q)
            else:
                x = "Question: " + q + '\n '
            
            if self.add_soft_prompts and self.split != 'train' and not self.plan_first and not self.plan_only:
                x += self.soft_prompt_text['prefix']
            
            self.x.append(x)

    def extract_answer(self, completion):
        match = self.ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.INVALID_ANS

    def is_correct(self, model_completion, gt_example):
        gt_answer = self.extract_answer(gt_example)
        print(gt_answer)
        assert gt_answer != self.INVALID_ANS
        try:
            pred_answer = self.extract_answer(model_completion)
            print(pred_answer)
        except:
            print(model_completion)
            pred_answer = self.INVALID_ANS
        return pred_answer == gt_answer
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return dict(x=self.x[idx], y=self.y[idx])



class GSMData(BaseData):

    def __init__(self, split: str, soft_prompt_text: list, 
                 invalid_ans="[invalid]", add_soft_prompts=False, 
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):
        super(GSMData, self).__init__(split, soft_prompt_text, 
            invalid_ans, add_soft_prompts, only_at_front, plan_first, plan_only,
            prompt_template, step_type_ids, tokenizer, step_type_predictor)

    def load_data(self, split: str):
        return load_dataset('gsm8k', 'main')[split]
    
    def parse_q_a(self, example):
        cot, ans = example["answer"].split("####")
        cot = cot.strip()
        cot = cot.split('. ')
        cot_steps = []
        for step in cot:
            for s in step.strip().split('\n'):
                s = s.strip()
                if len(s) == 0:
                    continue
                if s[-1] != '.':
                    s += '.'
                cot_steps.append(s)
            
        ans = ans.strip()
        q = example["question"].strip()
        inst = 'Solve the following problem step by step, and give a numerical answer.'
        return inst, q, cot_steps, ans

    def extract_step_type(self, q, cot_steps):
        step_types = [['prefix'] for _ in cot_steps]
        if self.step_type_predictor is not None:
            text = "Question: " + q + '\n ' + ' \n'.join(cot_steps) + '\n'
            start = len(q.split('\n'))
            new_step_types = self.step_type_predictor.predict(text, start)
            if len(new_step_types) != len(cot_steps):
                print(new_step_types)
                print(cot_steps)
            for i, t in enumerate(new_step_types):
                step_types[i].append(t)

        if self.step_type_ids is None:
            if self.step_type_re is not None:
                rgx = r"<<.*>>"
                for i, step in enumerate(cot_steps):
                    matches = re.findall(rgx, step)
                    if len(matches) > 0:
                        for match in matches:
                            step_types[i] += re.findall(self.step_type_re, match)
                            # print(match)
                            # print(step_types)
                    else:
                        step_types[i] += re.findall(self.step_type_re, step) 
        else:
            for i, step in enumerate(cot_steps):
                text_ids = self.tokenizer.encode(step)
                for j in self.step_type_ids:
                    if j in text_ids:
                        step_types[i].append(self.tokenizer.convert_ids_to_tokens(i))

        for i, step in enumerate(cot_steps):
            if len(step_types[i]) == 1:
                step_types[i].append('assignment')

        return step_types


    def clean_text(self, rgx_list, text):
        new_text = text
        for rgx_match in rgx_list:
            new_text = re.sub(rgx_match, '', new_text)
        return new_text

    

class AquaData(BaseData):
    def __init__(self, split: str, soft_prompt_text: list, 
                 invalid_ans="[invalid]", add_soft_prompts=False, 
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):
        super(AquaData, self).__init__(split, soft_prompt_text, 
            invalid_ans, add_soft_prompts, only_at_front, plan_first, plan_only,
            prompt_template, step_type_ids, tokenizer, step_type_predictor)

    def load_data(self, split: str):
        return load_dataset('aqua_rat', 'raw')[split]
    
    def parse_q_a(self, example):
        choice = "(" + "(".join(e.strip() for e in example["options"])
        choice = choice.replace("(", " (").replace(")", ") ")
        choice = "Answer Choices:" + choice

        q = example["question"].strip() + '\n ' + choice

        ans = example["correct"].strip()
        cot = example["rationale"].strip()
        cot = cot.split('. ')
        cot_steps = []
        for step in cot:
            for s in step.strip().split('\n'):
                s = s.strip()
                if len(s) == 0:
                    continue
                cot_steps.append(s)

        if 'Explanation' in cot_steps[0]:
            print(cot_steps[0])
            cot_steps[0] = cot_steps[0][12:].strip()

        if len(cot_steps[0]) < 2:
            cot_steps = cot_steps[1:]
        
        if len(cot_steps) == 0:
            return None, None, None, None
        
        if 'Ans' in cot_steps[-1] or 'ANS' in cot_steps[-1] or 'ans' in cot_steps[-1] or 'Option' in cot_steps[-1]:
            print(cot_steps[-1])
            if len(cot_steps[-1]) < 30:
                cot_steps = cot_steps[:-1]
        elif len(cot_steps[-1]) < 5:
            print(cot_steps[-1])
            cot_steps = cot_steps[:-1]

        inst = 'Solve the following problem step by step, and choose the best answer from the given choices.'

        return inst, q, cot_steps, ans

    def extract_answer(self, completion):
        preds = completion.split("The answer is:")
        pred = preds[-1].strip()
        pred = pred.upper()
        pred = re.findall(r'A|B|C|D|E', pred)
        if len(preds) > 1:
            if len(pred) > 0:
                return pred[0]
            else:
                return self.INVALID_ANS
        else:
            try:
                return pred[-1]
            except:
                print(preds)
                return self.INVALID_ANS
        

class MathData(BaseData):
    def __init__(self, split: str, soft_prompt_text: list, type=None,
                 invalid_ans="[invalid]", add_soft_prompts=False, 
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):
        self.type = type
        super(MathData, self).__init__(split, soft_prompt_text, 
            invalid_ans, add_soft_prompts, only_at_front, plan_first, plan_only,
            prompt_template, step_type_ids, tokenizer, step_type_predictor)

    def load_data(self, split: str):
        all_data = load_dataset('competition_math')[split]
        filtered_data = []
        if self.type in ['Algebra', 'Geometry', "Counting & Probability", 
                    "Intermediate Algebra", "Number Theory", "Prealgebra",
                    "Precalculus"]:
            for ex in all_data:
                if ex['type'] == self.type:
                    filtered_data.append(ex)
            return filtered_data
        else:
            return all_data
    
    def parse_q_a(self, example):

        q = example["problem"].strip()
        cot = example["solution"].strip()
        cot = cot.split('. ')
        cot_steps = []
        for step in cot:
            for s in step.strip().split('\n'):
                s = s.strip()
                if len(s) == 0:
                    continue
                cot_steps.append(s)

        ans = self.extract_answer(example["solution"])

        inst = 'Solve the following problem step by step, and give the exact answer.'

        return inst, q, cot_steps, ans

    def extract_answer(self, completion):
        if'The answer is:' in completion:
            pred = completion.split('The answer is:')[-1].strip()
        elif'The answer is ' in completion:
            pred = completion.split('the answer is ')[-1].strip()
        elif'the answer is ' in completion:
            pred = completion.split('the answer is ')[-1].strip()
        elif 'boxed' in completion:
            ans = completion.split('boxed')[-1]
            if (ans[0] == '{'):
                stack = 1
                a = ''
                for c in ans[1:]:
                    if (c == '{'):
                        stack += 1
                        a += c
                    elif (c == '}'):
                        stack -= 1
                        if (stack == 0): break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
            a = _strip_string(a)
            pred=a

        else:
            pattern = '-?\d*\.?\d+'
            pred = re.findall(pattern, completion)
            if(len(pred) >= 1):
                pred = pred[-1]
            else: pred = ''
        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]
            if pred[-1] == "/":
                pred = pred[:-1]
        pred=_strip_string(pred)
        if 'boxed' in pred:
            ans = pred.split('boxed')[-1]
            if (ans[0] == '{'):
                stack = 1
                a = ''
                for c in ans[1:]:
                    if (c == '{'):
                        stack += 1
                        a += c
                    elif (c == '}'):
                        stack -= 1
                        if (stack == 0): break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
            a = _strip_string(a)
            pred=a
        return pred

    def is_correct(self, model_completion, gt_example):
        gt_answer = self.extract_answer(gt_example)
        print(gt_answer)
        assert gt_answer != self.INVALID_ANS
        try:
            pred_answer = self.extract_answer(model_completion)
            print(pred_answer)
        except:
            print(model_completion)
            pred_answer = self.INVALID_ANS
        return compare_both_string_and_number_format(pred_answer, gt_answer)
    

class SVAMPData(BaseData):
    def __init__(self, split: str, soft_prompt_text: list, 
                 invalid_ans="[invalid]", add_soft_prompts=False, 
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):
        super(SVAMPData, self).__init__(split, soft_prompt_text, 
            invalid_ans, add_soft_prompts, only_at_front, plan_first, plan_only,
            prompt_template, step_type_ids, tokenizer, step_type_predictor)

    def load_data(self, split: str):
        return load_dataset('ChilleD/SVAMP')[split]
    
    def parse_q_a(self, example):

        q = example["Body"].strip() + example["Question"].strip()
        cot_steps = [example["Equation"].strip()]

        ans = str(example["Answer"])

        inst = 'Solve the following problem by an equation, and give a numerical answer.'

        return inst, q, cot_steps, ans

    def extract_answer(self, completion):
        pred = super().extract_answer(completion)
        return delete_extra_zero(pred)
    


class StrategyQAData(BaseData):

    def __init__(self, split: str, soft_prompt_text: dict, 
                 invalid_ans="[invalid]", add_soft_prompts=False, 
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):
        super(StrategyQAData, self).__init__(split, soft_prompt_text, 
            invalid_ans, add_soft_prompts, only_at_front, plan_first, plan_only,
            prompt_template, step_type_ids, tokenizer, step_type_predictor)
        
    def load_data(self, split: str):
        return load_dataset("ChilleD/StrategyQA")[split]

    def parse_q_a(self, example):
        inst = 'Answer the following question Ture or False step by step using the supporting facts in your knowledge.'
        question = example["question"]
        answer = example["answer"]
        answer = 'Ture' if answer else 'False'

        cot_steps = [sentence.strip() for sentence in example['facts'].split('.') if sentence.strip()]
        return inst, question, cot_steps, answer







class HotpotData(BaseData):
    def __init__(self, split: str, soft_prompt_text: dict, 
                 invalid_ans="[invalid]", add_soft_prompts=False, 
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):
        super(HotpotData, self).__init__(split, soft_prompt_text, 
            invalid_ans, add_soft_prompts, only_at_front, plan_first, plan_only,
            prompt_template, step_type_ids, tokenizer, step_type_predictor)
        
    def load_data(self, split: str):
        return load_dataset('hotpot_qa','fullwiki')[split]
    
    
    def parse_q_a(self, example):
        print(example)
        inst = 'Answer the following question step by step using the provided supporting facts.'
        question = example["question"].strip()
        answer = example["answer"].strip()
        rag_steps = []
        seen_steps = set()
        for step in example['supporting_facts']['title']:
            if step in seen_steps:
            # 如果检测到重复，打印整个 example
                print("Found duplicate step! Printing the entire example:")
                print(example)
            else:
                seen_steps.add(step)
                rag_steps.append(step)

        return inst, question, rag_steps, answer
    





















if __name__ == "__main__":


    data_class = StrategyQAData
    dataset = data_class("train", {})
    print(dataset)
   
    
