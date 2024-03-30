from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig,GenerationConfig
from loguru import logger
import json
import torch
from transformers import set_seed
import numpy as np
from scipy.stats import entropy
import time
import os
from dataset_preprocess import MimicDataset
set_seed(42)

def create_datasets():
    dataset_mimic = MimicDataset()
    preprocessed_data = dataset_mimic.process_test_data()
    return preprocessed_data

def formatting_prompts_func(examples):
    question = examples["question"]
    table_description = examples["table_description"]
    schema = examples["schema"]    
    prompt="""<s>[INST] ### Task\nGenerate a SQL query to answer [QUESTION]{question}[/QUESTION]\n\n ### Database Table Description\nThe table name and its corresponding description are as follows:\n{table_description}\n\n ### Database Schema\nThis query will run on a database whose schema is represented in this string:\n{schema} [INST] \n\n ### Answer\nGiven the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]\n[SQL]"""
    prompt=prompt.format(question=question,table_description=table_description,schema=schema)
    return prompt

def get_args():
    parser = ArgumentParser(description="LLM SELECTOR")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_directory", type=str)
    arguments = parser.parse_args()
    return arguments

def main(args):

    val_data = create_datasets()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto",torch_dtype=torch.float16)
    generation_config = GenerationConfig(
                temperature=0.1,
                top_p=0.75,
                top_k=40,
                num_beams=4,
                pad_token_id=tokenizer.pad_token_id,
            )
    device="cuda"

    whole_result={}
    start_time = time.time()
    for idx , data in enumerate(val_data):
        prompt = formatting_prompts_func(data)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=4096,

            )
    
    
        if hasattr(generation_output, 'beam_indices'):
                transition_scores = model.compute_transition_scores(generation_output.sequences, generation_output.scores,beam_indices=generation_output.beam_indices,normalize_logits=True)
        else:
            transition_scores = model.compute_transition_scores(generation_output.sequences, generation_output.scores,normalize_logits=True)
        transition_scores = transition_scores.to("cpu")
        
        output_length = inputs.input_ids.shape[1] + np.sum(transition_scores.numpy() < 0, axis=1)
        probabilities = torch.exp(transition_scores.sum(axis=1) / (output_length))
        
        
        logprobs = torch.exp(transition_scores[0]).numpy()

        mean_lowest25 = np.mean(sorted(logprobs)[:25])
        mean_highest25 = np.mean(sorted(logprobs)[-25:])
        maxp = np.max(logprobs)
        minp = np.min(logprobs)
        rangep = maxp - minp
        meanp = np.mean(logprobs)
        stdp = np.std(logprobs)
        
        entropyp = entropy(np.exp(logprobs))
        if stdp != 0:
            kurtosisp = np.mean((logprobs - meanp)**4) / stdp ** 4
            skewnessp = np.mean((logprobs - meanp)**3) / stdp ** 3
        else:
            kurtosisp = 0
            skewnessp = 0
        perplexityp = np.exp(-np.mean(logprobs))
        prob_scores=[]
        for i in range(len(transition_scores)):
            prob=torch.exp(transition_scores[i]).sum(axis=0)
            prob_score = prob.numpy()/len(transition_scores[i])
            prob_scores.append(prob_score)
        
        input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        generated_ids = generation_output.sequences[:, input_length:]
        preds = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
        result = {}
        result['question'] = data["question"]
        result['pred'] = preds
        result['db_id'] = data["id"]
        result['prob'] = probabilities.numpy().tolist()[0]
        result["prob_score"] = prob_scores[0]
        result["mean_lowest25"] = mean_lowest25
        result["mean_highest25"] = mean_highest25
        result["maxp"] = maxp
        result["minp"] = minp
        result["rangep"] = rangep
        result["stdp"] = stdp
        result["entropyp"] = float(entropyp)
        result["kurtosisp"] = kurtosisp
        result["skewnessp"] = skewnessp
        whole_result[data["id"]] = result
        logger.info(f"Completed index {idx}")
            
    class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        
    end_time = time.time()
    print(f"Duration : {end_time-start_time}")
    inference_result_path = args.output_directory
    output_file = "prediction.json"
    os.makedirs(inference_result_path, exist_ok=True)
    out_file = os.path.join(inference_result_path, output_file)
    with open(out_file, 'w') as f:
        json.dump(whole_result, f, cls=NpEncoder)
    print("completed")
    
if __name__ == "__main__":
    args = get_args()
    main(args)
    