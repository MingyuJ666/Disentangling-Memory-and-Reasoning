from openai import OpenAI
import re
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_data.preprocess import *
api = ''


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


    try:
        parsed_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Print detailed error message for debugging
        print(f"JSONDecodeError: {e.msg}. Error at line {e.lineno}, column {e.colno}.")
        # Show 40 characters before and after the error position
        print(f"Problematic JSON snippet: {json_str[e.pos-40:e.pos+40]}")
        raise

    return parsed_dict


def planning(question):
    template = f"""
<Question>
{question}
<\Question>


provide a reasoning planning for the above question, each step in your reasoning plan should be in the following format:

*Step name*: 
# put the name of step here.
**Requirement**: 
# If this step needs to be reasoning, return [reason], if this step needs factual knowledge return [rag].
**Knowledge based**:  
# Only if this step needs factual knowledge，put a query in question sentences about this factual knowledge for retrieval.
**Content**: 
# If this step is about reasoning, please provide your reasoning thinking, if this step needs factual knowledge please provide factual knowledge.

    """
    return template


def NER_agent(questions, model_name="gpt-4o"):
    template = f"""
      Please use your knowldge to briefly answer below questions:
<Questions>
{questions}
<\Questions>

Your answer format should strictly be in following steps:
```json
{{
      "The content of question 1": "The answer of question 1",
....
}}
```


      """
    text = get_response(model_name, template)

    return text


def get_label(input_string):
    planing = get_response("gpt-4o", planning(input_string))
    list = clean_and_parse_json_string_with_codeblock(NER_agent(extract_knowledge_based(planing)))
    for item in list.keys():
        planing = planing.replace(item, item + " [rag]" + list[item])
    return extract_labeled_content_as_list(planing)



def generate_StrategyQA_agent(type):

    data = load_dataset("ChilleD/StrategyQA")[type]
    dict = []
    json_file = "dataset_folder/StrategyQA_{}.json".format(type)
    

    for example in data:
        question = example["question"]
        answer = example["answer"]
        answer = 'True' if answer else 'False'

        max_retries = 5
        retry_count = 0
        cot_steps = None

        while retry_count < max_retries:
            try:
                cot_steps = get_label(question)
                break  
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

        processed_data = []

        for entry in cot_steps:
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
    



   
if __name__ == '__main__':
    generate_StrategyQA_agent("train")
    clean_json("dataset_folder/StrategyQA_train.json", "dataset_folder/StrategyQA_train_clean.json")

    
    
