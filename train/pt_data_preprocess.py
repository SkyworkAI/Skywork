#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from itertools import chain
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False, trust_remote_code=True)

    def tokenize_function(examples):
        output = tokenizer(examples[args.text_field])
        return output
    
    block_size = args.block_size
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    filename = '.'.join(args.input_file.split("/")[-1].split(".")[:-1])
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, filename)
    tmp_cache_dir = os.path.join(args.output_dir, filename+"_text")

    if args.data_type == "jsonl":   
        raw_dataset = load_dataset("json", data_files=args.input_file, cache_dir=tmp_cache_dir, keep_in_memory=False, encoding="utf8")
    elif args.data_type == 'text':
        raw_dataset = load_dataset("text", data_files=args.input_file, cache_dir=tmp_cache_dir, keep_in_memory=False, encoding="utf8")
    else:
        raise NotImplementedError(f"data type should be in json,txt not {args.data_type}")

    print("remove_column_names:", raw_dataset.column_names['train'])
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_dataset.column_names['train'],
        load_from_cache_file=True,
        keep_in_memory=False,
        cache_file_names = {k: os.path.join(tmp_cache_dir, 'tokenized.arrow') for k in raw_dataset},
        desc="Running tokenizer on dataset",
    )
    if args.filter_by_length is not None:
        tokenized_dataset["train"] = tokenized_dataset["train"].filter(
            lambda x: len(x["input_ids"]) >= args.filter_by_length
        )
    grouped_datasets = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        keep_in_memory=False,
        cache_file_names = {k: os.path.join(tmp_cache_dir, 'grouped.arrow') for k in tokenized_dataset},
        desc=f"Grouping texts in chunks of {block_size}",
    )
    processed_dataset = grouped_datasets
    processed_dataset.save_to_disk(cache_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tokenizer_path', default=None, type=str, required=True)
    parser.add_argument('-w', '--preprocessing_num_workers', default=64, type=int)
    parser.add_argument('-b', '--block_size',default=4096,type=int)
    parser.add_argument('-i', '--input_file',default=None, type=str,help="")
    parser.add_argument('-o', '--output_dir',default=None, type=str,help="")
    parser.add_argument('--data_type',default='jsonl', type=str,help="")
    parser.add_argument('--text_field',default='text', type=str,help="")
    parser.add_argument('--filter_by_length',default=None, type=int, help="")

    args = parser.parse_args()

    main(args)
