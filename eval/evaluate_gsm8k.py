# origin code from https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_gsm8k.py
import re
import torch
import argparse
import jsonlines
import numpy as np
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

ans_re1 = re.compile(r"(\-?[0-9][0-9\.\,]*)")

ans_re2 = re.compile(r'=\s*(\$?-?[0-9][0-9\.\,]*)')

prefix_sky1 = 'answer is'
prefix_sky2 = '答案是'

INVALID_ANS = "[invalid]"

def get_match_str(match, idx):
    match_str = match[idx]
    match_str = match_str.replace(",", "")
    if match_str.endswith('.'):
        match_str = match_str[:-1]
    if match_str.endswith('.00'):
        match_str = match_str[:-3]
    if match_str.endswith('.0'):
        match_str = match_str[:-2]
    return match_str

def doc_to_text(doc):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(
            tokens[raw_text_len:])
        sents.append(sent)
    return sents

def generate_sample(model, tokenizer, input_txt):
    input_ids = tokenizer([input_txt], padding=False)["input_ids"]
    context_enc = torch.tensor(input_ids, device=model.device)
    raw_text_len = len(input_ids[0])
    print(f"Input text: {input_txt}\n")
    outputs = model.generate(context_enc, pad_token_id=tokenizer.pad_token_id)
    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    print(f"\nOutput text: {output_text}\n")
    return output_text


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS

def extract_answer(text):
    if prefix_sky1 in text:
        text = text.split(prefix_sky1)[-1]
    if prefix_sky2 in text:
        text = text.split(prefix_sky2)[-1]
    match1 = re.findall(ans_re1, text)
    match2 = re.findall(ans_re2, text)
    ans = []
    if match1:
        match_str1 = get_match_str(match1, -1)
        ans.append(match_str1)
    if match2:
        match_str2 = get_match_str(match2, -1).replace('$','')
        ans.append(match_str2)
    if len(ans) > 0:
        return eval(ans[-1])
    else:
        return INVALID_ANS

def is_correct(completion, answer):
    completion = completion.split('<|endoftext|>')[0]
    completion = completion.split('\n\n\n')[0]
    completion = completion.split("\n\n")[0]
    completion = completion.split("Question:")[0]

    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    clear_answer = extract_answer(completion)
    return clear_answer == gold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="/data/shared/public/liang.zhao/skywork-13b-models/skywork-13b-base/",
    )
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="gsm8k_res.jsonl"
    )

    args = parser.parse_args()

    fewshot_prompt = open("./eval/gsm8k_prompt.txt").read()
    if args.sample_input_file is not None:
        dataset = load_from_disk(args.sample_input_file)
    else:
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = load_dataset("gsm8k", "main", download_config=config)

    test = dataset["test"]

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, padding_side='left'
    )
    if "qwen" in args.checkpoint_path.lower():
        tokenizer.pad_token = '<|extra_0|>'
        tokenizer.eos_token = '<|endoftext|>'
    else:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, device_map="auto", trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model.generation_config.do_sample = False

    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
    tot_length = test.num_rows
    acc_res = []
    for doc in test:
        context = doc_to_text(doc)
        completion = generate_sample(model, tokenizer, context)
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        doc["completion"] = completion
        doc["acc"] = acc
        f_output.write(doc)
        acc_res.append(acc)

    f_output.close()
    print(acc_res)
    print("Acc: ", np.mean(acc_res))

# 'acc=True': 723
# Acc=723/1319=0.548
