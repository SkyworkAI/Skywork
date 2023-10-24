from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import AutoConfig
import torch
import argparse


def load(tokenizer_path, checkpoint_path, use_cpu=False):
    print('Loading tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False, trust_remote_code=True, padding_side='left')
    tokenizer.add_tokens("[USER]")
    tokenizer.add_tokens("[BOT]")
    tokenizer.add_tokens("[SEP]")

    print('Loading model ...')
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    if use_cpu:
        device_map = "cpu"
    else:
        device_map = "balanced_low_0"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, config=config, device_map=device_map, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(
        checkpoint_path, trust_remote_code=True)

    model.generation_config.do_sample = True
    model.eval()

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    return model, tokenizer


def extract_res(response):
    if "[BOT]" in response:
        response = response.split("[BOT]")[1]
    else:
        response = response[prompt_len+1:]
    if "<s>" in response:
        response = response.split("<s>")[-1]
    if "</s>" in response:
        response = response.split("</s>")[0]
    if "[SEP]" in response:
        response = response.split("[SEP]")[0]
    return response[1:]



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Skywork-cli-demo")
    parser.add_argument("-m", "--model_path", type=str, default="skywork-13b-chat")
    parser.add_argument("-n", "--max_new_tokens", type=int, default=1000)
    parser.add_argument("-t", "--temperature", type=float, default=0.95)
    parser.add_argument("-p", "--top_p", type=float, default=0.8)
    parser.add_argument("-k", "--top_k", type=int, default=5)
    parser.add_argument("--cpu", action='store_true', help="inference with cpu")

    args = parser.parse_args()

    model, tokenizer = load(args.model_path, args.model_path, args.cpu)

    while True:

        doc = input("输入：")
        prompts = "</s><s>[USER]" + doc + "[SEP][BOT]<s>"
        input_tokens = tokenizer(prompts, return_tensors='pt').to(model.device)

        response = model.generate(**input_tokens,
                                 max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id,
                                 do_sample=True,
                                 top_p=args.top_p,
                                 top_k=args.top_k,
                                 temperature=args.temperature,
                                 num_return_sequences=1,
                                 repetition_penalty=1.1,
                                 bos_token_id=1,
                                 eos_token_id=2)
        response = tokenizer.decode(response.cpu()[0], skip_special_tokens=True)
        response = extract_res(response)
        print("模型输出：")
        print(response)