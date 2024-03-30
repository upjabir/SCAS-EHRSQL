from dataset_preprocess import MimicDataset
import os
from loguru import logger
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer,DataCollatorWithPadding,TrainingArguments, Trainer, TrainerCallback
from accelerate import Accelerator
from argparse import ArgumentParser
import evaluate
import numpy as np
import torch
from copy import deepcopy
from peft import LoraConfig, TaskType,get_peft_model,prepare_model_for_kbit_training
import bitsandbytes as bnb
os.environ["WANDB_PROJECT"] = "classification" # log to your project

def get_args():
    parser = ArgumentParser(description="Fine-tune an LLM sequence model")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_checkpoints/classification_model_3_epoch",)
    parser.add_argument("--model_name", type=str, default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of epochs"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--log_steps",type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    
    parser.add_argument("--use_int4", action="store_true", default=False)
    parser.add_argument("--use_int8", action="store_true", default=False)
    parser.add_argument("--disable_lora", action="store_true", default=False)
    parser.add_argument("--all_linear", action="store_true", help="Use Lora on all linear layers", default=False)
    parser.add_argument("--long_lora", action="store_true", help="Use long lora settings", default=False)
    parser.add_argument("--rope_scale", type=float,default=None)
    
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    arguments = parser.parse_args()
    return arguments

def add_prompt(example):
    example["prompt"] ="<s>[INST] SQL Schema:\n{schema}\nQuestion:\n{question}[/INST]</s>".format(schema=example["schema"],question=example["question"])
    return example

def create_dataset(tokenizer):
    data_class = MimicDataset()
    dataset = data_class.process_data()
    dataset = dataset.class_encode_column("numerical_label")
    dataset= dataset.train_test_split(train_size=0.9, seed=42,stratify_by_column="numerical_label")
    dataset["val"] = dataset.pop("test")
    dataset = dataset.map(add_prompt)
    
    pos_weights = len(dataset['train'].to_pandas()) / (2 * dataset['train'].to_pandas().numerical_label.value_counts()[1])
    neg_weights = len(dataset['train'].to_pandas()) / (2 * dataset['train'].to_pandas().numerical_label.value_counts()[0])
    
    max_len = 0
    for set_name in dataset:
        for idx in range(len(dataset[set_name])):
            len_ = len(tokenizer.encode(dataset[set_name][idx]["prompt"]))
            if len_ > max_len:
                max_len = len_

    tokenize_func = lambda example: tokenizer(example['prompt'], truncation=True, max_length=max_len)
    dataset =dataset.map(tokenize_func).remove_columns(["prompt","id","question","label","schema","table_description"])
    dataset = dataset.rename_column("numerical_label", "label")
    dataset.set_format("torch")
    
    return dataset, pos_weights,neg_weights

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

def load_model_and_tokenizer(model_name, args,use_eos_as_pad=True):
    
    added_tokens=False
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)

    if use_eos_as_pad:
        logger.info("Uses eos token as pad")
        tokenizer.pad_token = tokenizer.eos_token
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_index = Accelerator().process_index
    model = AutoModelForSequenceClassification.from_pretrained(model_name,trust_remote_code=True,device_map={"": device_index},use_cache=False)
    
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
            task_type=TaskType.SEQ_CLS, 
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
            model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=True if args.gradient_checkpointing else False)

        logger.info("Getting PEFT model...")
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
    return tokenizer,model

def get_weighted_trainer(pos_weight, neg_weight):
    
    class _WeightedBCELossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 2 labels with different weights)
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weight, pos_weight], device=labels.device, dtype=logits.dtype))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    return _WeightedBCELossTrainer

def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric= evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
        
def main(args):
    """
    Training function
    
    """
    tokenizer,model = load_model_and_tokenizer(args.model_name, args,use_eos_as_pad=True)
    dataset ,pos_weights,neg_weights = create_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        lr_scheduler_type= "cosine",
        warmup_ratio= 0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch", #epoch
        save_strategy="epoch", #epoch
        load_best_model_at_end=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=True,
        report_to="wandb",
        run_name="codellama_seq_cls",
        logging_strategy="steps",
        max_grad_norm= 0.3
    )

    weighted_trainer = get_weighted_trainer(pos_weights, neg_weights)
    collator = DataCollatorWithPadding(tokenizer)
    trainer = weighted_trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset["val"],
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(CustomCallback(trainer))
    trainer.train()
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
    
    logger.info("DONE")

if __name__ == "__main__":
    args = get_args()
    main(args)