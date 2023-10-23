#!/usr/bin/env python
# coding=utf-8

import numpy as np
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path
import datasets
import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm 
import transformers
from transformers import AutoTokenizer
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM
)
import tensor_parallel as tp
import sys 
import gc
import argparse
from functools import partial
import os 
import random 
from transformers.trainer_utils import set_seed
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
import pickle
device = "cuda"

def compute_loss(tokenized_texts, attention_mask, model, tokenizer, add_start_token=False):

    loss_func = CrossEntropyLoss(reduction="none")

    if add_start_token:
        tokenizer.bos_token = getattr(tokenizer, "bos_token", "<s>") 
        tokenizer.bos_token_id = getattr(tokenizer, "bos_token_id", len(tokenizer)-101 ) 
        bos_tokens_tensor = torch.tensor(
            [[tokenizer.bos_token_id]] * tokenized_texts.size(dim=0))
        tokenized_texts = torch.cat(
            [bos_tokens_tensor, tokenized_texts], dim=1).to(device)
        attention_mask = torch.cat(
            [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64), attention_mask], dim=1
        ).to(device)
    else:
        tokenized_texts = tokenized_texts.to(device)
        attention_mask = attention_mask.to(device)

    labels = tokenized_texts[:, 1:]

    with torch.no_grad():
        logits = model(tokenized_texts, attention_mask=attention_mask).logits
        logits = logits[:, :-1]
        attention_mask = attention_mask[:, :-1]
        loss = loss_func(logits.transpose(1, 2), labels) * attention_mask

    num_tokens = torch.sum(attention_mask).item() 
    return torch.sum(loss).item(), num_tokens

def load_model_tokenizer_config():
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from transformers.generation import GenerationConfig
    config = AutoConfig.from_pretrained(args.model_path,  trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", config=config, trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)
    while config.num_attention_heads % args.n_gpus != 0:
        args.n_gpus //= 2
        args.batch_size //= 2 
    return model, tokenizer, config 

def main():

    model, tokenizer, config = load_model_tokenizer_config()

    tokenizer.padding_size = "right"
    if "qwen-14b" in args.model_path.lower():
        tokenizer.pad_token = '<|extra_0|>'
        tokenizer.eos_token = '<|endoftext|>'
        args.batch_size = 1 
        args.n_gpus = 1 
    else:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if args.data_type is not None:
        raw_dataset = datasets.load_dataset(args.data_type, data_files=args.dataset, split=args.split if len(args.split) > 0 else None)
    else:
        raw_dataset = datasets.load_dataset(args.dataset, name=args.subset if len(args.subset) > 0 else None, split=args.split if len(args.split) > 0 else None)
    input_texts = raw_dataset[args.input_text_field]
    if args.max_tokens is not None:
        input_texts = [text[:args.max_tokens//3] for text in input_texts] 

    if args.max_samples is not None:
        input_texts = input_texts[:args.max_samples] 

    print("input_texts numbers is", len(input_texts))
    torch.cuda.empty_cache()

    model = tp.tensor_parallel(model, [i for i in range(args.n_gpus)])

    total_loss = 0
    total_tokens = 0
    for i in tqdm(range(0, len(input_texts), args.batch_size), total = len(input_texts) // args.batch_size):
        start_idx = i 
        end_idx = min(i + args.batch_size, len(input_texts))
        batch_texts = input_texts[start_idx:end_idx]
        tokenized_texts = tokenizer(batch_texts, add_special_tokens=False, padding=True, truncation=True, max_length=args.max_tokens, return_tensors="pt")

        loss, num_tokens = compute_loss(tokenized_texts=tokenized_texts["input_ids"], attention_mask=tokenized_texts["attention_mask"], model=model, tokenizer=tokenizer, 
                                add_start_token=False)
        total_loss += loss 
        total_tokens += num_tokens

    with open(args.output_file, "w", encoding="utf8") as f:
        avg_loss = total_loss / total_tokens
        f.write(f"{avg_loss:.4f}\t{total_tokens}\n")
        print(f"{avg_loss:.4f}\t{total_tokens}\n")

    
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, default="")
    parser.add_argument("-d", "--dataset", type=str, default="emozilla/pg19")  #tau/scrolls gov_report, emozilla/
    parser.add_argument("-s", "--subset", type=str, default=None) # gov_report
    parser.add_argument("-i", "--input-text-field", type=str, default="text")
    parser.add_argument("-b", "--batch-size", type=int, default="batch size")
    parser.add_argument("-o", "--output-file", type=str)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--data-type", type=str, default=None)
    parser.add_argument("--n-gpus", type=int, default=None)
    parser.add_argument("--aggressive-memory", action="store_true")
    parser.add_argument("--split", type=str, default="train")

    args = parser.parse_args()
    set_seed(1234)

    main()