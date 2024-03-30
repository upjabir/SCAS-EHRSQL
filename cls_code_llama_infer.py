from transformers import AutoTokenizer
from dataset_preprocess import MimicDataset
import torch
from peft import PeftConfig, AutoPeftModelForSequenceClassification
import numpy as np
from loguru import logger
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Fine-tune an LLM sequence model")
    parser.add_argument("--model_path", type=str)
    arguments = parser.parse_args()
    return arguments

def add_prompt(example):
    example["prompt"] ="<s>[INST] SQL Schema:\n{schema}\nQuestion:\n{question}[/INST]</s>".format(schema=example["schema"],question=example["question"])
    return example

def main(args):
    data_class = MimicDataset()
    dataset = data_class.process_test_data()
    dataset = dataset.map(add_prompt)

    peft_model_id = args.model_path
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoPeftModelForSequenceClassification.from_pretrained(peft_model_id, 
                                                                            quantization_config=None, 
                                                                            device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    device="cuda"


    model.eval()
    preds = []
    for idx , item in enumerate(dataset):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        preds.append(outputs.logits.to("cpu").numpy().astype(np.float32))
        logger.info(f"Completed{idx}")
    np.save('data_test.npy', preds)
    
if __name__ == "__main__":
    args = get_args()
    main(args)