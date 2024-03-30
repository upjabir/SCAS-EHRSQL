import json
import numpy as np
from dataset_preprocess import MimicDataset
from openai_wrapper import ChatModel
import openai
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from argparse import ArgumentParser
from dotenv import load_dotenv
load_dotenv()

voter_template="""
Based on the database schema and table description, determine which AI assistant's answer accurately identifies whether the given question can generate an SQL query or not
### Database Table Description
The table name and its corresponding description are as follows:
{table_description}

### Database Schema
This query will run on a database whose schema is represented in this string:
{schema}

### Instructions
- Do not hallucinate.
- Use have two options to answer. 1) Able to generate SQL Query. 2) Unable to generate SQL Query
- Use this json format for answering final answer .

```json
{{
    "Final Answer":  \\ Wheather able to generate SQL Query or not.
}}
```
{few_shots}
Question: "{question}"
Ai Assitant 1's Answer: {model1_answer}
Ai Assitant 2's Answer: {model2_answer}
Answer: Let's think step by step.

"""

format_template="""
### Rewrite the text by following correct format.
### Format
- Use have two options to answer. 1) Able to generate SQL Query. 2) Unable to generate SQL Query
```json
{{
    "Final Answer":  \\ Wheather able to generate SQL Query or not.
}}

text: "{text}"
Remember, Always follow format when returning.
"""

def check_format(text,model):
    text_list=[]
    text_list.append(text)
    for i in range(3):
        search_result = re.search(r'"Final Answer": "(.*?)"', text_list[i], re.DOTALL)
        if search_result:
            return search_result.group(1)
        format_prompt = format_template.format(text=text_list[i])
        messages = [{"role": "user", "content": format_prompt}]
        res = model.complete(message=messages,max_tokens=8192,stop=None)
        text_list.append(res.text)

    return "Unable to generate"

def generate_schema_few_shots(data,indeces):
    str_data=""
    for idx in indeces:
        question = data[int(idx)]["question"]
        num_label= data[int(idx)]["numerical_label"]
        if num_label ==1:
            label = "Able to generate SQL Query."
        else:
            label="Unable to generate SQL Query."
        str_data+=f"Question: '{question}'\nAnswer: {label}\n\n"
    return str_data

def write_json(path, file):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w+') as f:
        json.dump(file, f)

def get_args():
    parser = ArgumentParser(description="LLM SELECTOR")
    parser.add_argument("--code_llama_result_path", type=str)
    parser.add_argument("--custom_tokenizer_result_path", type=str)
    parser.add_argument("--submission_result_path", type=str)
    arguments = parser.parse_args()
    return arguments

def main(args):
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    codellam_cls = np.load(args.code_llama_result_path)
    codellam_cls = np.concatenate(codellam_cls)[:, 1]
    codellam_cls_list=[]
    for item in codellam_cls:
        if item > 1.2:
            codellam_cls_list.append("Able to generate SQL Query.")
        else:
            codellam_cls_list.append("Unable to generate SQL Query.")
            

    with open(args.custom_tokenizer_result_path) as f:
        codellama_skip_null_data = json.load(f)
        
    codellama_skip_null_list=[]
    for key,value in codellama_skip_null_data.items():
        if value ==1:
            codellama_skip_null_list.append("Able to generate SQL Query.")
        else:
            codellama_skip_null_list.append("Unable to generate SQL Query.")


    model = ChatModel(engine="gpt35turbo0613_16k", temperature=0, request_timeout=20)
    dataset_mimic = MimicDataset()
    preprocessed_data = dataset_mimic.process_data()
    val_data = dataset_mimic.process_test_data()

    embed_model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        trust_remote_code=True,
    )
    print("loaded models")
    train_question_embeddings = embed_model.encode(preprocessed_data["question"])
    print("completed train question embeddings")
    
    final_dict={}
    for idx , data in enumerate(val_data):
        print(f"Executing{idx}")
        val_question = data["question"]
        val_question_embeddings = embed_model.encode([val_question])
        similarity_with_questions = cosine_similarity(val_question_embeddings, train_question_embeddings)
        
        model1_answer = codellama_skip_null_list[idx]
        model2_answer = codellam_cls_list[idx]
        
        top_answerable_indices = similarity_with_questions.argsort()[0][-4:][::-1]
        few_shot_schema_data = generate_schema_few_shots(preprocessed_data,top_answerable_indices)
        schema_generation_prompt = voter_template.format(table_description=data["table_description"],
                                                        schema=data["schema"],
                                                        few_shots=few_shot_schema_data,
                                                        question=val_question,
                                                        model1_answer=model1_answer,
                                                        model2_answer=model2_answer)
        
        messages = [{"role": "user", "content": schema_generation_prompt}]
        try:
            res = model.complete(message=messages,max_tokens=8192,stop=None)
            new_out = check_format(res.text,model)
            final_dict[data["id"]]=new_out
        except Exception as e:
            print("Error occurred:", e)
            print(data["id"])
            final_dict[data["id"]]="Unable to generate"

    write_json(args.submission_result_path,final_dict)

if __name__ == "__main__":
    args = get_args()
    main(args)