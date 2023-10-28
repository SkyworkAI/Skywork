from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import AutoConfig
import torch


def load(tokenizer_path, checkpoint_path):
    print('Loading tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False, trust_remote_code=True, padding_side='left')
    tokenizer.add_tokens("[USER]")
    tokenizer.add_tokens("[BOT]")
    tokenizer.add_tokens("[SEP]")

    print('Loading model ...')
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, config=config, device_map="balanced_low_0", torch_dtype=torch.bfloat16, trust_remote_code=True)
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
    if "<s>" in response:
        response = response.split("<s>")[-1]
    if "</s>" in response:
        response = response.split("</s>")[0]
    if "[SEP]" in response:
        response = response.split("[SEP]")[0]
    return response

def special_encode(prompt, tokenizer):
       raw_str = "[USER]%s[SEP][BOT]" % prompt.strip().replace("\r", "")
       bos_id = tokenizer.bos_token_id
       eos_id = tokenizer.eos_token_id
       sep_id = tokenizer.encode("[SEP]")[-1]
       res_id = [eos_id, bos_id]
       arr = raw_str.split("[SEP]")
       for elem_idx in range(len(arr)):
              elem = arr[elem_idx]
              elem_id = tokenizer.encode(elem)[1:]
              res_id += elem_id
              if elem_idx < len(arr) - 1:
                     res_id.append(sep_id)
       return res_id

if __name__ == '__main__':
    tokenizer_path='skywork/skywork-13b-chat'
    checkpoint_path = 'skywork/skywork-13b-chat'
    model, tokenizer = load(tokenizer_path, checkpoint_path)

    doc = "写一首七言绝句"
    input_tokens = special_encode(doc, tokenizer)
    input_tokens = torch.tensor(input_tokens).to(model.device).reshape(1, -1)
    response = model.generate(input_tokens,
                                 max_new_tokens=1000,
                                 pad_token_id=tokenizer.pad_token_id,
                                 do_sample=True,
                                 top_p=0.8,
                                 top_k=5,
                                 temperature=0.95,
                                 num_return_sequences=1,
                                 repetition_penalty=1.1,
                                 bos_token_id=1,
                                 eos_token_id=2)
    response = tokenizer.decode(response.cpu()[0], skip_special_tokens=True)
    response = extract_res(response)
    print(response)

    """生成结果：
    千里莺啼绿水滨，
    万家歌笑白云间。
    男女老少皆朋友，
    和谐社会见温馨。 
    """

    doc = "我是一名运动员，最近比赛取得很好的成绩受到大家的关注和认可。帮我写一份微博文案，帮我感谢大家支持我，要有日常感，并语言随意一些"
    input_tokens = special_encode(doc, tokenizer)
    input_tokens = torch.tensor(input_tokens).to(model.device).reshape(1, -1)
    
    response = model.generate(input_tokens,
                                 max_new_tokens=1000,
                                 pad_token_id=tokenizer.pad_token_id,
                                 do_sample=True,
                                 top_p=0.8,
                                 top_k=5,
                                 temperature=0.95,
                                 num_return_sequences=1,
                                 repetition_penalty=1.1,
                                 bos_token_id=1,
                                 eos_token_id=2)
    response = tokenizer.decode(response.cpu()[0], skip_special_tokens=True)
    response = extract_res(response)
    print(response)

    """生成结果：
    
    谢谢每一个在我运动生涯中陪伴我的人，你们的支持、鼓励和信任，是我前进的动力。这段时间的比赛，让我感受到了前所未有的成就和喜悦，它们不仅属于我，更属于那些一路陪伴我成长的人们。

    从心底里感谢所有支持我的人，是你们的支持，让我有了勇气和力量去追求更高的目标。每一次的鼓励，都是我前进道路上的光芒，每一次的点赞，都是对我努力的肯定。

    生活中的点滴，让我们相互理解，相互支持，一起走过了无数个黎明和黄昏。感谢你们的陪伴，让我不再孤单，让我更加坚定地走向未来。

    未来的路还很长，但我愿意陪你一起走下去。因为我知道，你们的支持，会一直伴随着我前行。

    再次感谢所有支持我的人，你们是我心中最亮的星。

    #运动员# #体育精神# 
    """