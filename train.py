import os
import random
import numpy as np
import wandb
import torch
from dotenv import load_dotenv

from functools import partial
from datasets import load_dataset
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from metric import compute_metrics

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True #False
    np.random.seed(seed)
    random.seed(seed)
    
from datasets import load_dataset
dataset_name = "kjchoi/news_summ-data"
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")
dataset = load_dataset(dataset_name, use_auth_token=HUGGINGFACE_KEY)

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v2', use_fast=True)
config = AutoConfig.from_pretrained("gogamza/kobart-base-v2")
model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2", config=config).to(device)
config.max_length = 128
config.num_beams = 1

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors='pt')

def tokenize_function(examples, tokenizer, max_input_length, max_target_length):
    bos_token = tokenizer.bos_token
    inputs = [bos_token + ' ' + doc for doc in examples['contents']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

prep_fn = partial(tokenize_function, tokenizer=tokenizer, max_input_length=1024, max_target_length=128)
tokenized_dataset = dataset.map(prep_fn,
                                batched=True,
                                num_proc=4,
                                remove_columns=dataset['train'].column_names,
                                load_from_cache_file=True
                            )

train_data = tokenized_dataset['train']
val_data = tokenized_dataset['validation']

# Metrics
metric_fn = partial(compute_metrics, tokenizer=tokenizer)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir = os.path.join(".."), # local path
    logging_dir = os.path.join(".."),
    num_train_epochs = 3,
    save_steps = 10000,
    eval_steps = 5000,
    logging_steps = 1000,
    evaluation_strategy = 'steps',
    per_device_train_batch_size = 8, # 16,
    per_device_eval_batch_size = 8, #16,
    warmup_steps=2000,
    weight_decay=1e-4,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    overwrite_output_dir=False,
    predict_with_generate=True
)

# Trainer
trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=metric_fn
)

def main():
    load_dotenv(dotenv_path='path.env')
    WANDB_AUTH_KEY = os.getenv('WANDB_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    wandb_name = "kobart_summarization_test"
    wandb.init(
        entity="kjchoi",
        project="Korean-Abstractive-Summarization-Test",
        name=wandb_name,
        group='kobart_finetuning')
    
    print('\nTraining')
    trainer.train()
    wandb.finish()
    
if __name__=="__main__":
    seed_everything(42)
    main()