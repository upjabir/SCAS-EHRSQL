import argparse
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,TrainingArguments,AutoConfig
from datasets import Dataset
import torch
import os
from peft import LoraConfig, TaskType,get_peft_model,prepare_model_for_kbit_training
import pandas as pd
import math
import bitsandbytes as bnb
import transformers
from typing import Dict
from typing import List, Optional
from accelerate import Accelerator
import numpy as np
import random
from dataset_preprocess import MimicDataset
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="defog/sqlcoder-7b-2")
    parser.add_argument("--split_model", action="store_true",default=False)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="./model_checkpoints/codellama_filter_schema")
    parser.add_argument("--epochs",  type=float,default=3)
    parser.add_argument("--log_steps",type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--trust_remote_code", action="store_true", default=True) 
    parser.add_argument("--use_int4", action="store_true", default=False)
    parser.add_argument("--use_int8", action="store_true", default=False)
    parser.add_argument("--disable_lora", action="store_true", default=False)
    parser.add_argument("--all_linear", action="store_true", help="Use Lora on all linear layers", default=False)
    parser.add_argument("--long_lora", action="store_true", help="Use long lora settings", default=False)
    parser.add_argument("--rope_scale", type=float,default=None)
    parser.add_argument("--custom_target", type=str, help="Custom target for lora",default=None)
    parser.add_argument("--pad_token_id", default=None, type=int, help="The end of sequence token.")
    parser.add_argument("--use_eos_as_pad", action="store_true",help="Use eos token as pad token.",default=False)
    parser.add_argument("--seed",default=42,type=int,help="Seed for random number generators")
    parser.add_argument("--completion_only", default=False,action="store_true", help="Only use completion loss")
    parser.add_argument("--constant_length_dataset", default=False,action="store_true", help="Only use constant length dataset")
    return parser.parse_args()

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
    
def create_datasets(args):
    dataset_mimic = MimicDataset()
    preprocessed_data = dataset_mimic.process_data(skip_null=False)
    preprocessed_data = preprocessed_data.shuffle(seed=args.seed)    
    print(f"Size of the train set: {len(preprocessed_data)}")
    return preprocessed_data

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
        print(special_tokens_dict)
    else:
        return False
    num_new_tokens = len(list(special_tokens_dict.keys())) + (0 if custom_tokens is None else len(custom_tokens))
    logger.info(f"Number of new tokens: {num_new_tokens}")
    if len(list(special_tokens_dict.keys())) > 0:
        tokenizer.add_special_tokens(special_tokens_dict)
    if custom_tokens is not None:
        tokenizer.add_tokens(custom_tokens,special_tokens=True)

    model.resize_token_embeddings(len(tokenizer))    
    return True

def find_all_linear_names(args, model,add_lm_head=True):
    cls = bnb.nn.Linear4bit if args.use_int4 else (bnb.nn.Linear8bitLt if args.use_int8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if add_lm_head and not "lm_head" in lora_module_names:
        logger.info("Adding lm_head to lora_module_names")
        lora_module_names.add("lm_head")

    return list(lora_module_names)

def get_model_config(args):
    config_kwargs = {
        "trust_remote_code": True if args.trust_remote_code else None,
    }
    config = AutoConfig.from_pretrained(args.model_name, **config_kwargs)

    config.use_cache = False
    if not args.gradient_checkpointing:
        logger.info("Not using gradient checkpointing")
        config.gradient_checkpointing = False
    else:
        logger.info("Using gradient checkpointing")
        config.gradient_checkpointing = True

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.seq_length > orig_ctx_len and args.rope_scale is None:
        scaling_factor = float(math.ceil(args.seq_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        logger.info("Scaling context length by %f", scaling_factor)
    elif args.rope_scale is not None:
        scaling_factor = float(math.ceil(args.rope_scale))
        logger.info("Scaling context length by %f", scaling_factor)
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    else:
        logger.info("Not scaling context length")
    return config

def setup_tokenizer_and_model(args,model_config):
    if args.split_model:
        logger.info("Splitting the model across all available devices...")
        device_index = Accelerator().process_index
        device_map = {"": device_index}
        kwargs = {"device_map":device_map}
    else:
        kwargs = {"device_map":None}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              trust_remote_code=args.trust_remote_code,
                                              use_fast=True)

    if args.pad_token_id is not None:
        logger.info("Using pad token id %d", args.pad_token_id)
        tokenizer.pad_token_id = args.pad_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(args.pad_token_id)
        
    if args.use_eos_as_pad:
        logger.info("Uses eos token as pad")
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    if args.use_int4:
        logger.info("Using int4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        optimizer = "adamw_bnb_8bit"
        args.use_int8 = False
    elif args.use_int8:
        logger.info("Using int8 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        optimizer = "adamw_bnb_8bit"
    else:
        logger.info("Using no quantization")
        bnb_config = None
        optimizer = "adamw_torch"
        
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 quantization_config=bnb_config,
                                                 trust_remote_code=args.trust_remote_code,
                                                 config=model_config,
                                                 **kwargs)
    
    added_tokens = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model
    )

    if not args.disable_lora and args.lora_alpha is None:
        args.lora_alpha = args.lora_rank * 2
        logger.info("Lora alpha set to None... Setting lora_alpha to %d", args.lora_alpha)
    
    if not args.disable_lora and args.all_linear:
        target_modules = find_all_linear_names(args, model)
        logger.info("Using LORA on all linear layers: %s", target_modules)
        if added_tokens:
            target_modules.pop(target_modules.index("lm_head"))
            logger.info("Removing lm_head from target modules, will use in modules_to_save")
    elif not args.disable_lora and args.custom_target:
        target_modules = args.custom_target.split(",")
        logger.info("Using LORA on all linear layers: %s", target_modules)
        if added_tokens:
            target_modules.pop(target_modules.index("lm_head"))
            logger.info("Removing lm_head from target modules, will use in modules_to_save")
            
    elif not args.disable_lora:
        target_modules = None
        logger.info("Using LORA on default layers")

    if not args.disable_lora:
        if args.long_lora:
            logger.info("Using long lora settings...")
            modules_to_save = ["embed_tokens","input_layernorm","post_attention_layernorm","norm"]

            if added_tokens:
                logger.info("Adding lm_head to modules_to_save")
                modules_to_save.append("lm_head")
        elif added_tokens:
            modules_to_save =  ["embed_tokens","lm_head"]
        else:
            modules_to_save = None
            
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save
        )
        logger.info("Using LORA...")
        if args.use_int4 or args.use_int8:
            logger.info("Preparing model for kbit training...")
            model = prepare_model_for_kbit_training(model,
                                                    use_gradient_checkpointing=True if args.gradient_checkpointing else False)

        logger.info("Getting PEFT model...")
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
    else:
        logger.info("Using Full Finetuning")
        
    return model,tokenizer,optimizer

def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["question"])):
        question = examples["question"][i]
        table_description = examples["table_description"][i]
        schema = examples["schema"][i]
        sql_query=examples["label"][i]
    
        prompt="""<s>[INST] ### Task\nGenerate a SQL query to answer [QUESTION]{question}[/QUESTION]\n\n ### Database Table Description\nThe table name and its corresponding description are as follows:\n{table_description}\n\n ### Database Schema\nThis query will run on a database whose schema is represented in this string:\n{schema} [INST] \n\n ### Answer\nGiven the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]\n[SQL]{sql}[\SQL]</s>"""
        prompt=prompt.format(question=question,table_description=table_description,schema=schema,sql_query=sql_query)
        output_text.append(prompt)

    return output_text



if __name__ == "__main__":
    args = get_args()
    print(args)
    seed_all(args.seed)
    model_config = get_model_config(args)
    
    model,tokenizer,optimizer = setup_tokenizer_and_model(args,model_config)
    train_dataset = create_datasets(args)
    print(f"Size of the train set: {len(train_dataset)}")
    def get_warmup_steps(num_training_steps, warmup_ratio=0.05):
        return math.ceil(num_training_steps * warmup_ratio)

    training_args = TrainingArguments(
        do_train=True,
        do_eval=False, 
        output_dir=args.output_dir,
        dataloader_drop_last=False,
        save_strategy="epoch", #epoch
        logging_strategy="steps",
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        optim=optimizer,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        weight_decay=args.weight_decay,
        bf16=False,
        fp16=True,
        report_to="wandb",
        run_name="codellama_finetune_epoch3_new",
        ddp_find_unused_parameters=False,
        max_grad_norm= 0.3
    )
    print(training_args)
    if args.completion_only:
        logger.info("Using completion only loss...")
        logger.warning("Make sure to manually set this value in the code to a list of ids")
        response_template =  "### Answer"
        assert response_template is not None, "Response template must be set"
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,tokenizer=tokenizer
        )
        packing = False
        logger.info("Using data collator for completion only")
    elif args.constant_length_dataset:
        logger.info("Using constant length dataset with packing")
        data_collator =None
        packing=True
    else:
        logger.info("Using No data collator and packing")
        data_collator = None 
        packing = None
        
    print("data collator", data_collator)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func= formatting_prompts_func,
        max_seq_length=args.seq_length,
        tokenizer=tokenizer,
        data_collator=data_collator,
        packing=packing,
        neftune_noise_alpha=5)
    trainer.train()
    
    logger.info("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
    logger.info("DONE")