import torch
from peft import PeftModel
import os
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,AutoConfig
import argparse

import transformers
from typing import Dict
import json
from typing import List, Optional

from loguru import logger

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    custom_tokens:Optional[List[str]]=None,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    if len(list(special_tokens_dict.keys())) >0 or custom_tokens is not None:
        logger.info("Resizing tokenizer and embedding...")
        logger.info(f"Special tokens dict: {special_tokens_dict}")
        logger.info("Custom tokens: %s", custom_tokens)
    else:
        return False
    num_new_tokens = len(list(special_tokens_dict.keys())) + (0 if custom_tokens is None else len(custom_tokens))
    logger.info(f"Number of new tokens:{num_new_tokens}")
    if len(list(special_tokens_dict.keys())) > 0:
        tokenizer.add_special_tokens(special_tokens_dict)
    if custom_tokens is not None:
        tokenizer.add_tokens(custom_tokens,special_tokens=True)

    model.resize_token_embeddings(len(tokenizer))
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="codellama/CodeLlama-7b-Instruct-hf",help="Give the name of the model as it appears on the HuggingFace Hub")
    parser.add_argument("--adapter_path", type=str, default="./model_checkpoints/codellama_filter_schema/final_checkpoint",help="Give the path to the Lora model")
    parser.add_argument("--output", type=str, default="./merged_models/codellama_filter_schema",help="Give the path to the output folder")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--context_size", type=int, default=None, help="Context size during fine-tuning")
    parser.add_argument("--custom_tokens",type=str,default=None)
    parser.add_argument("--pad_token_id",type=int,default=None)

    
    args = parser.parse_args()
    args.output = os.path.realpath(args.output)
    adapter_path = os.path.realpath(args.adapter_path)

    if args.base_model is not None:
        base_model_name = args.base_model
        logger.info("Using base model %s", base_model_name)
    else:
        adapter_config_path = os.path.join(adapter_path,"adapter_config.json")
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config["base_model_name_or_path"]
        logger.info("Base model not given, using %s", base_model_name)

    if args.cpu:
        device_map = {"": "cpu"}
        logger.info("Using CPU")
        logger.warning("This will be slow, use GPUs with enough VRAM if possible")
    else:
        device_map = "auto"
        logger.info("Using Auto device map")
        logger.warning("Make sure you have enough GPU memory to load the model")


    os.makedirs("offload", exist_ok=True)

    base_config = AutoConfig.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name   
    )   

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        logger.info("added special pad token ")
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        offload_folder="offload", 
        trust_remote_code=True,
        quantization_config=None
    )
    
    added_tokens = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=base_model,
        custom_tokens=None
    )
    logger.info("Loading Lora model...")
        
    lora_model = PeftModel.from_pretrained(
        base_model, 
        adapter_path, 
        torch_dtype=torch.float16,
        device_map=device_map,
        offload_folder="offload", 

    )
    print(lora_model)

    os.makedirs(args.output, exist_ok=True)
    logger.info("Merging model...")
    lora_model = lora_model.merge_and_unload()
    logger.info(f"Merge complete, saving model to {args.output} ...")
    print(lora_model)
    lora_model.save_pretrained(args.output)
    logger.info("Model saved")

    tokenizer.save_pretrained(args.output)