import parser
from openai import OpenAI
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_data.preprocess import *
# fill in your openai api key
api =''
from tqdm import tqdm
import argparse
import random


def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = ""
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def extract_labeled_content_as_list(input_string):
    # Splitting the input string by step names
    steps = re.split(r'\*Step name\*:', input_string)

    labeled_content = []
    for step in steps:
        if step.strip():
            # Extracting requirement
            requirement_match = re.search(r'\*\*Requirement\*\*: \[(.*)\]', step)
            if requirement_match:
                requirement = f"[{requirement_match.group(1)}]"

            # Extracting content
            content_match = re.search(r'\*\*Content\*\*: (.*)', step)
            if content_match:
                content = content_match.group(1).strip()

                # Adding labeled content to the list
                labeled_content.append(f"{requirement}: {content}")

    return labeled_content


def get_response(model_name, prompt):
    messages = [{"role": "user", "content": f"""{prompt}"""}]
    print("GPT loading {}...".format(model_name))
    client = OpenAI(api_key=api)
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return completion.choices[0].message.content


def extract_knowledge_based(text):
    # Define a pattern to match the **knowledge based** sections
    pattern = r"\*\*Knowledge based\*\*:(.*?)\*\*Content\*\*"
    matches = re.findall(pattern, text, re.DOTALL)

    # Clean and return the extracted knowledge based sections
    return [match.strip() for match in matches]


def clean_and_parse_json_string_with_codeblock(json_str):

    json_str = json_str.replace('```json', '').replace('```', '')

    json_str = re.sub(r'"[^"]*"\s*:\s*""\s*,?', '', json_str)

    json_str = re.sub(r',\s*}', '}', json_str)  
    json_str = re.sub(r',\s*]', ']', json_str)  

    json_str = json_str.strip().strip(',')

    if json_str.count('{') != json_str.count('}') or json_str.count('[') != json_str.count(']'):
        raise ValueError("The JSON string has unbalanced braces or brackets.")


    print("Cleaned JSON String:\n", json_str)



    parsed_dict = json.loads(json_str)


    return parsed_dict


def planning(question, answer):
    template = f"""
Here is the question:
<Question>
{question}
<\Question>

Here is the correct answer:
<Correct Answer>
{answer}
<\Correct Answer>

Factual knowledge aligns with objective reality and can be verified through evidence or observation, such as scientific facts or historical events.

provide an inference planning for the above question to get the correct answer, you should first generate all [rag] steps, then generate all [reason] steps, each step in your inference plan must adhere strictly to the following format:

*Step name*: 
# put the name of the step here.
**Requirement**: 
# If this step needs reasoning, return "[reason]" as the label, if this step needs factual knowledge return "[rag]" as the label.
**Knowledge based**:  
# Only if this step needs factual knowledge, put a query in question sentences about this factual knowledge for retrieval.
**Content**: 
# If this step is about reasoning, please provide your reasoning thinking, if this step needs factual knowledge please provide factual knowledge.

    """
    return template


def NER_agent(questions, model_name="gpt-4o"):
    template = f"""
    Factual knowledge is information that aligns with objective reality and can be verified through evidence or observation, such as scientific facts or historical events.

    Please provide factual knowledge for below question set:
<Questions>
{questions}
<\Questions>

You should return a dic in json format, for each element in dic, the key is each question in <Questions>, the value is the Factual knowledge of each question in <Questions>.
Your answer format should strictly be in following steps:
```json
{{
      "question 1": "The factual knowledge of question 1",
....
}}
```

      """
    text = get_response(model_name, template)

    return text


def get_label(input_string,answer):
    planing = get_response("gpt-4o", planning(input_string,answer))
    list = clean_and_parse_json_string_with_codeblock(NER_agent(extract_knowledge_based(planing)))
    for item in list.keys():
        planing = planing.replace(item, item + " [rag]" + list[item])
    return extract_labeled_content_as_list(planing)



def generate_StrategyQA_agent(type):

    data = load_dataset("ChilleD/StrategyQA")[type]
    dict = []
    json_file = "dataset_folder/StrategyQA_{}.json".format(type)
    

    for example in tqdm(data):
        question = example["question"]
        answer = example["answer"]
        answer = 'True' if answer else 'False'

        max_retries = 5
        retry_count = 0
        cot_steps = None

        while retry_count < max_retries:
            try:
                cot_steps = get_label(question,answer)
                if(len(cot_steps) > 0):
                    print('success, aoligei')
                    break
                    
                else:
                    retry_count +=1
                    print(f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")
            except Exception as e:
                retry_count += 1
                print(f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")
        
        # if the maximum number of retries is reached, skip the question
        if retry_count == max_retries:
            print(f"Skipping question due to repeated errors: {question}")
            continue

        
        new_entry = {
            "question": question,
            "answer": answer,
            "cot_steps": cot_steps,  
            "split": type
        }
        dict.append(new_entry)
       


        with open(json_file, "w") as f:
           json.dump(dict, f, indent=4)


def generate_MMLU_pro_agent(split):
    type = split
    if type =="train":
        type = "validation"
    else:
        type = "test"
    print(type)
    data = load_dataset("TIGER-Lab/MMLU-Pro")[type]
    dict = []
    json_file = "dataset_folder/MMLU_Pro_{}.json".format(type)

    for example in tqdm(data):
        question = format_example(example["question"],example["options"])
        answer = example["answer"]

        max_retries = 5
        retry_count = 0
        cot_steps = None

        while retry_count < max_retries:
            try:
                cot_steps = get_label(question, answer)
                if (len(cot_steps) > 0):
                    break
                else:
                    retry_count += 1
                    print(
                        f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")
            except Exception as e:
                retry_count += 1
                print(
                    f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")

        # if the maximum number of retries is reached, skip the question
        if retry_count == max_retries:
            print(f"Skipping question due to repeated errors: {question}")
            continue


        new_entry = {
            "question": question,
            "answer": answer,
            "cot_steps": cot_steps,
            "split": type
        }
        dict.append(new_entry)

        with open(json_file, "w") as f:
           json.dump(dict, f, indent=4)


def generate_CommensenQA_agent(split):
    type = split
    if type =="train":
        type = "train"

        data = load_dataset("tau/commonsense_qa")[type]
        dict = []
        json_file = "dataset_folder/commonsense_qa_{}.json".format(type)
        

        for example in tqdm(data):
            q = example['question']
            choices = example['choices']['text']
            labels = example['choices']['label']
            question = f"Question: {q} Options: "
            for label, choice in zip(labels, choices):
                question += f"{label}.{choice} "
            answer = example['answerKey']

            max_retries = 5
            retry_count = 0
            cot_steps = None

            while retry_count < max_retries:
                try:
                    cot_steps = get_label(question, answer)
                    if (len(cot_steps) > 0):
                        break
                    else:
                        retry_count += 1
                        print(
                            f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")
                except Exception as e:
                    retry_count += 1
                    print(
                        f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")

            # if the maximum number of retries is reached, skip the question
            if retry_count == max_retries:
                print(f"Skipping question due to repeated errors: {question}")
                continue


            new_entry = {
                "question": question,
                "answer": answer,
                "cot_steps": cot_steps,
                "split": type
            }
            dict.append(new_entry)

            with open(json_file, "w") as f:
                json.dump(dict, f, indent=4)

    else:  
        type = "validation"
        data = load_dataset("tau/commonsense_qa")[type]
        print(data)
        json_file = "dataset_folder/commonsense_qa_test_clean_CC.json"
        dict = []
        for example in tqdm(data):
            q = example['question']
            choices = example['choices']['text']
            labels = example['choices']['label']
            question = f"Question: {q} Options: "
            for label, choice in zip(labels, choices):
                question += f"{label}.{choice} "
            answer = example['answerKey']

            new_entry = {
                "question": question,
                "answer": answer,
                "cot_steps": [],
                "split": 'test'
            }
            dict.append(new_entry)
        with open(json_file, "w") as f:
            json.dump(dict, f, indent=4)
        exit()

        

   


def clean_json(json_file, json_file1):
    with open(json_file, "r") as f:
        data = json.load(f)

    list = []
    pattern = re.compile(r"\[(.*?)\]")

    for item in data:

        question = item['question']
        answer = item['answer']
        cot_steps = item['cot_steps']
        type = item['split']

        filtered_statements = [statement for statement in cot_steps if ': ' in statement and statement.split(': ')[1].strip()]
        if not filtered_statements:
            continue
        

        processed_data = []

        for entry in filtered_statements:
            match = pattern.search(entry)
            if match:
                tag = match.group(1)
                if tag not in ['reason','rag']:
                    entry = entry.replace(f"[{tag}]", "[rag]")  
            processed_data.append(entry)
        

        new_entry = {
            "question": question,
            "answer": answer,
            "cot_steps": processed_data,  
            "split": type
        }

        list.append(new_entry)
    
    with open(json_file1, "w") as f:
        json.dump(list, f, indent=4)
    


def generate_truthfulqa_agent(split):
    type = split
    if type =="train":
        type = "train"

        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
        json_file = "dataset_folder/truthful_qa_{}.json".format(type)
        data = ds['validation']
        choice_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        train_data = data.select(range(int(len(data) * 0.8)))
        dict = []
        for example in train_data:
            question_text = example['question']
            choices = example['mc1_targets']['choices']
            labels = example['mc1_targets']['labels']
            choices_and_labels = list(zip(choices, labels))
            random.shuffle(choices_and_labels) 
            shuffled_choices, shuffled_labels = zip(*choices_and_labels)
            formatted_choices = [f"{choice_letters[i]}. {choice}" for i, choice in enumerate(shuffled_choices)]
            question = f"{question_text} {' '.join(formatted_choices)}"
            answer_index = shuffled_labels.index(1)
            answer = choice_letters[answer_index]
            

            max_retries = 5
            retry_count = 0
            cot_steps = None
            

            while retry_count < max_retries:
                try:
                    cot_steps = get_label(question, answer)
                    if (len(cot_steps) > 0):
                        break
                    else:
                        retry_count += 1
                        print(
                            f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")
                except Exception as e:
                    retry_count += 1
                    print(
                        f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")

            # if the maximum number of retries is reached, skip the question
            if retry_count == max_retries:
                print(f"Skipping question due to repeated errors: {question}")
                continue


            new_entry = {
                "question": question,
                "answer": answer,
                "cot_steps": cot_steps,
                "split": type
            }
            dict.append(new_entry)

            with open(json_file, "w") as f:
                json.dump(dict, f, indent=4)

    else:  
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
        json_file = "dataset_folder/truthful_qa_{}_clean_CC.json".format(type)
        data = ds['validation']
        test_data = data.select(range(int(len(data) * 0.8), len(data)))
        dict = []
        choice_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
       
        for example in test_data:
            question_text = example['question']
            choices = example['mc1_targets']['choices']
            labels = example['mc1_targets']['labels']
            choices_and_labels = list(zip(choices, labels))
            random.shuffle(choices_and_labels) 
            shuffled_choices, shuffled_labels = zip(*choices_and_labels)
            formatted_choices = [f"{choice_letters[i]}. {choice}" for i, choice in enumerate(shuffled_choices)]
            question = f"{question_text} {' '.join(formatted_choices)}"
            answer_index = shuffled_labels.index(1)
            answer = choice_letters[answer_index]
            
            new_entry = {
                "question": question,
                "answer": answer,
                "cot_steps": [],
                "split": 'test'
            }
            dict.append(new_entry)
            with open(json_file, "w") as f:
                json.dump(dict, f, indent=4)
        exit()


def main(args):
    os.makedirs("dataset_folder", exist_ok=True)
    if args.dataset ==  "StrategyQA":
        generate_StrategyQA_agent(args.mode)
    elif args.dataset == "MMLU_Pro":
        generate_MMLU_pro_agent(args.mode)
    elif args.dataset == "commonsense_qa":
        generate_CommensenQA_agent(args.mode)
    elif args.dataset == "truthful_qa":
        generate_truthfulqa_agent(args.mode)
        print(args.mode)
        
    file = os.path.join("dataset_folder", "{}_{}.json".format(args.dataset, args.mode))
    clean_file = os.path.join("dataset_folder", "{}_{}.json".format(args.dataset, args.mode + "_clean_CC"))
    clean_json(file, clean_file)


   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='truthful_qa', choices = ['commonsense_qa',"StrategyQA","truthful_qa"])
    parser.add_argument('--mode',type=str,default="train",choices = ['train','test'])
    args = parser.parse_args()
    main(args)

    
    
