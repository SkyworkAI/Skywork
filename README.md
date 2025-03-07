
<!-- <div align="center">
<h1>
  ✨Skywork
</h1>
</div> -->
<div align="center"><img src="misc/skywork_logo.png" width="550"/></div>

<p align="center">
🤗 <a href="https://huggingface.co/Skywork" target="_blank">Hugging Face</a> • 🤖 <a href="https://modelscope.cn/organization/Skywork" target="_blank">ModelScope</a> • 👾 <a href="https://wisemodel.cn/organization/Skywork" target="_blank">Wisemodel</a> • 💬 <a href="https://github.com/SkyworkAI/Skywork/blob/main/misc/wechat.png?raw=true" target="_blank">WeChat</a>• 📜<a href="http://arxiv.org/abs/2310.19341" target="_blank">Tech Report</a>
</p>

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/SkyworkAI/Skywork)](https://github.com/SkyworkAI/Skywork/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/SkyworkAI/Skywork)](https://github.com/SkyworkAI/Skywork/fork)
</div>

<div align="center">

</div>


# Project Introduction
We are pleased to announce the open source release of the Skywork large-scale models. Skywork is a series of large models developed by the Kunlun Group · Skywork team. The models being open sourced this time include the **Skywork-13B-Base** model, **Skywork-13B-Chat** model, **Skywork-13B-Math** model, and **Skywork-13B-MM** model, as well as quantized versions of each model to support deployment and inference on consumer-grade GPUs.

Our open-source Skywork series models can be used for commercial purposes, but you need to follow our agreement and refrain from engaging in harmful activities. The characteristics of the Skywork open-source project are:：

- **Skywork-13B-Base**: The model was trained on a high-quality cleaned dataset consisting of **3.2 trillion** multilingual data (mainly Chinese and English) and code. It has demonstrated the best performance among models of similar scale in various evaluations and benchmark tests.

- **Skywork-13B-Chat**: The model has powerful conversational abilities, and we have further enhanced it in the field of creative writing. We have constructed a high-quality dataset of over ten thousand instructions and fine-tuned the model on ten specific creative writing tasks, enabling it to achieve results similar to ChatGPT in these tasks. Additionally, we open-source a benchmark consisting of approximately 500 samples for these 10 creative writing tasks.

- **Skywork-13B-Math**: This model has undergone specialized training to enhance its mathematical abilities. In the 13B-scale model, the Skywork-13B-Math model ranked 1st in the GSM8K benchmark, and it also performed exceptionally well on the MATH and CMATH benchmarks, placing it among the top-level 13B models.

- **Skywork-13B-MM**:  This is a multimodal model that allows users to utilize image information for tasks like Q&A and dialogue. 

- **Skywork/Skypile-150B**: This dataset is a collection of high-quality data extracted from Chinese web pages through our carefully curated data processing pipeline. The size of this open-source dataset is approximately 600GB, with a total token count of around 150 billion. It is one of the largest publicly available Chinese datasets.

- In addition, we have also disclosed the evaluation methods, data distribution studies, and training infrastructure optimization plans used in training the Skywork-13B model. We hope that these open-source materials can further inspire the community's understanding of large-scale model pre-training and drive the realization of Artificial General Intelligence (AGI).

If you are interested in more training and evaluation details, please refer to our [technical report](http://arxiv.org/abs/2310.19341), [Skymath]((https://arxiv.org/skywork-tech-report)) paper and [SkyworkMM](https://github.com/will-singularity/Skywork-MM/blob/main/skywork_mm.pdf) paper.
 
# News and Updates
* 2023.12.7 Our SkyPile-150B dataset is now accessible via [huggingface](https://huggingface.co/datasets/Skywork/SkyPile-150B). 
  
* 2023.11.2 We have uploaded the evaluation data we built, [MOCK_GSM8K_TEST](https://huggingface.co/datasets/Skywork/mock_gsm8k_test), and the Chinese domain evaluation data [ChineseDomainModelingEval](https://huggingface.co/datasets/Skywork/ChineseDomainModelingEval) to huggingface. If you need to evaluate LLMs, please download our evaluation dataset.

* 2023.10.31 Our technical report [Skywork: A More Open Bilingual Foundation Model](http://arxiv.org/abs/2310.19341) is available on arXiv, which includes more detailed evaluation methods, result comparisons, and technical details.

* 2023.10.30  We release the **Skywork-13B-Base** and  **Skywork-13B-Math** models, as well as quantized versions of each model to support deployment and inference on consumer-grade GPUs. We  open-source the Skywork/Skypile-150B dataset. This dataset contains over 150 billion high-quality tokens cleaned from Chinese web pages, making it the largest open-source Chinese dataset currently known.


# Table of contents

- [☁️Download URL](#Download-URL)
- [👨‍💻Model Introduction](#Model-Introduction)
- [🏆Model Evaluation](#Model-Evaluation)
- [📕Quickstart](#Quickstart)
- [📣Chat Model Output Examples](#Chat-Model-Output-Examples)
- [🚀Quantization](#Quantization)
- [🛫Fine-tuning](#Fine-tuning)
- [🍀Community and Ecosystem](#Community-and-Ecosystem)
- [⚠️Declaration and License Agreement](#Declaration-and-License-Agreement)
- [🤝Contact Us and Citation](#Contact-Us-and-Citation)


# Download URL
## Download URL of Skywork Models

|         | HuggingFace Base Model   | HuggingFace Quantized Model |  ModelScope Base Model   | ModelScope Quantized Model |  Wisemodel Base Model   | Wisemodel Quantized Model |
|:-------:|:-----------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
| **Skywork-13B-Base**      | 🤗 [Skywork-13B-Base](https://huggingface.co/Skywork/Skywork-13B-Base) | 🤗 [Skywork-13B-Base-8bits](https://huggingface.co/Skywork/Skywork-13B-Base-8bits) | 🤖[Skywork-13B-Base](https://www.modelscope.cn/models/skywork/Skywork-13B-Base) | 🤖 [Skywork-13B-Base-8bits](https://www.modelscope.cn/models/skywork/Skywork-13B-Base-8bits) |👾[Skywork-13B-Base](https://wisemodel.cn/models/Skywork/Skywork-13B-Base) | 👾 [Skywork-13B-Base-8bits](https://wisemodel.cn/models/Skywork/Skywork-13B-Base-8bits) |
| **Skywork-13B-Chat**      | 🤗coming soon | 🤗coming soon | 🤖coming soon | 🤖coming soon |👾coming soon | 👾coming soon |
| **Skywork-13B-Math**      | 🤗 [Skywork-13B-Math](https://huggingface.co/Skywork/Skywork-13B-Math) | 🤗 [Skywork-13B-Math-8bits](https://huggingface.co/Skywork/Skywork-13B-Math-8bits) | 🤖 [Skywork-13B-Math](https://www.modelscope.cn/models/skywork/Skywork-13B-Math) | 🤖 [Skywork-13B-Math-8bits](https://www.modelscope.cn/models/skywork/Skywork-13B-Math-8bits) |👾[Skywork-13B-Math](https://wisemodel.cn/models/Skywork/Skywork-13B-Math) | 👾 [Skywork-13B-Math-8bits](https://wisemodel.cn/models/Skywork/Skywork-13B-Math-8bits) |
| **Skywork-13B-MM**      | 🤗coming soon | - | 🤖coming soon | - |👾coming soon | - |

## Download URL of Skypile
|    Data    |    Download URL | 
|:-------:|:-----------:|
| Skywork/Skypile-150B |  [Hugging Face URL](https://huggingface.co/datasets/Skywork/SkyPile-150B)  |



## Download of Intermediate Model Checkpoints

We have also open-sourced the Skywork-13B-Base model and provided the model checkpoints trained on 500B, 1T, 1.5T, 2T, 2.5T, 3T and 3.1T tokens for community research into the evolution process of large language model capabilities.

| Model | Download URL |
| --------- | ------ | 
| Skywork-13B-Base-Intermediate     |  🤗[Skywork-13B-Base-Intermediate](https://huggingface.co/Skywork/Skywork-13B-Base-Intermediate)|
| Skywork-13B-Base-3.1T     |  🤗[Skywork-13B-Base-3.1T](https://huggingface.co/Skywork/Skywork-13B-Base-3.1TB)|


# Skywork-13B Introduction

## Training Data
We have developed a data cleaning pipeline with great care to effectively clean and filter low-quality data and eliminate harmful information from text data. Our Skywork-13B-Base model is trained on a  dataset with 3.2T tokens that consists of high-quality Chinese, English, and code data, all of which have been thoroughly cleaned. The English data comprises 52.2% of the dataset, the Chinese data accounts for 39.6%, and the code data makes up 8%. This comprehensive approach ensures optimal performance for both Chinese and English while also maintaining the ability to handle code.
|             | Category         | Percentage |
|-------------|------------------|------------|
| **English** | Webpages         | 39.8%      |
|             | Books            | 3.6%       |
|             | Academic Papers  | 3.0%       |
|             | Encyclopedia     | 0.5%       |
|             | Miscellany       | 2.9%       |
| **Chinese** | Webpages         | 30.4%      |
|             | Social Media     | 5.5%       |
|             | Encyclopedia     | 0.8%       |
|             | Miscellany       | 3.1%       |
| **Other Lang.**    | Encyclopedia           | 2.4%       | 
| **Code**    | Github           | 8.0%       | 




## Model Structure
Compared to the Llama-2-13B model, the Skywork-13B model adopts a relatively thinner and deeper network structure with 52 layers. At the same time, the FFN Dim and Hidden Dim are reduced to 12288 and 4608, respectively, to ensure that the model has a similar number of parameters as the original Llama-2-13B model. Based on our preliminary experimental results, a relatively thinner and deeper network structure can achieve better generalization performance under large batch size training. The detailed comparison between the Skywork-13B and Llama-2-13B models is as follows:


| Model Structure         | Llama-2-13B | Skywork-13B | 
|----------------------|:----:|:-----------:|
| Vocab. Size  | 32,000 |    65,536     | 
| Hidden Dim.  | 5,120 |    4,608     | 
| FFN Dim.  | 13,696 |    12,288     |
| Head Dim. | 128 |    128     | 
| Num. Heads | 40 |    36     | 
| Num. Layers | 40 |    52     | 
| Seq. Len. | 4,096 |    4,096     | 
| Positional Embedding | RoPE | RoPE |


## Tokenizer 
We use Byte-Pair Encoding (BPE) to tokenize the data, with a vocabulary size of 65536. Among them, there are 32000 Latin characters and subwords, 8000 Chinese characters and Unicode symbols, 25519 Chinese words, and the remaining 17 are reserved words.

| Category                            | Size    |
|---------------------------------|--------|
| Latin based words & subwords                 | 32000  |
| Chinese characters & Unicode symbols               | 8000   |
| Chinese words                        | 25519  |
| Reserved symbols                       | 17     |
| **Total**                         | **65536** |


## Training Methods
In order to make more precise use of data, we adopt a two-stage training method. In the first stage, we use general corpora to train the model's general abilities. In the second stage, we incorporate STEM (Science, Technology, Engineering, Mathematics) related data to further enhance the model's reasoning, mathematical, and problem-solving abilities.

### First-stage Pretraining
During the training process, we monitor the changes in model training loss and various abilities. The following figure shows the change curves of important indicators selected during the first stage of pre-training. The first stage of pre-training consists of two consecutive training processes, which are represented by different colors. The model completed in the first stage of pre-training is referred to as Skywork-13B-3.1T-Base.
![Alt text](misc/stage1_metrics.png)

### Second-stage Pretraining
In the second stage of pre-training, STEM-related data is added to the general language corpus for further training. The second stage training involves approximately 130 billion tokens, resulting in a total training of 3.2 T across both stages, and yielding our final Skywork-13B-Base model.

<img src="misc/stage2_ceval.png" alt="Image" width="500" height="400">

## Skypile-150B

### Introduction
Skypile-150B is a large dataset specifically designed for pre-training Chinese language models. It is constructed using publicly available web page data from the Chinese internet. The dataset has undergone extensive filtering to remove duplicate and harmful text. Additionally, advanced models like FastText and Bert have been employed to further refine the dataset and eliminate low-quality data.

### Language and Data Format
Skypile-150B dataset is a collection of Chinese data. The pages contain processed and cleaned text, in JSONL format. Each line represents a document, parsed using JSON. The text is stored in the "text" field.

### Sensitive information and bias
Although it has undergone strict cleaning and filtering, since it is built on a publicly accessible webpage established by Skypile-150B, it may still contain some sensitive information such as email addresses, phone numbers, or IP addresses. Therefore, users need to be careful and perform necessary additional filtering and cleaning before using the data.

### License Agreement
The use of data must comply with our License and must not be used for any purpose that poses a threat to national and social security or violates the law.

# Model Evaluation

## Documentation Perplexity Evaluation
The main goal of training a language model is to improve the accuracy of predicting the next word. With this in mind, we believe that evaluating the ability of a language model to generate articles in different domains is a crucial way to assess the performance of large-scale models. During model training, the likelihood of predicting the next word is typically measured using the Cross Entropy loss function. The overall loss function is calculated as the average of the losses when predicting the correct word at each position, which can be represented as:

```math
loss = -\sum^{n}_{i=1} log(p_i) / n = -log( \prod_{i=1}^n p_i) / n
```

Where $`n`$ is the length of the document, i.e., the number of tokens, and $`p_i`$ is the probability of the label word at position $i$. We know that the product of the probabilities of the label words at each position in the document is equal to the probability of generating that document. In this way, we connect the loss with the probability of generating the document. Since different models use different tokenizers and have different numbers of tokens, we multiply the loss function by the number of tokens $`n`$. This way, we only consider the part related to the probability of generating the article, and different models can be compared. We normalize the loss and convert it to perplexity by taking the exponential, making the differences between models more pronounced. For readability, the terms "loss" and "ppl" mentioned later refer to the normalized loss and perplexity of the model.

Based on the analysis above, we have chosen several hundred to thousands of high-quality articles that were published after september 1, 2023  across various fields. We have manually verified these articles to ensure their quality. It is important to note that none of the test data used in evaluating the Skywork model or any other models is included in their training set. Furthermore, the test data is diverse and of high quality, making it challenging for the models to gain an unfair advantage.

The figure below displays the performance of different open source models. Skywork-13B-Base achieves the best results.

|                  | Tech  | Movie | Gov.  | Game  | Finance | General | Average |
|------------------|-------|-------|-------|-------|---------|---------|---------|
| MOSS-7B          | 20.83 | 39.66 | 11.08 | 31.24 | 10.59   | 13.25   | 18.50   |
| InternLM-7B      | 13.43 | 24.90 | 5.88  | 19.78 | 6.17    | 8.10    | 11.17   |
| Qwen-7B          | 13.39 | 25.16 | 5.55  | 19.26 | 5.76    | 7.78    | 10.83   |
| Baichuan2-7B     | 12.89 | 23.26 | 5.34  | 18.36 | 5.68    | 7.62    | 10.41   |
| LLaMA2-13B       | 23.26 | 50.66 | 18.09 | 32.52 | 14.85   | 16.55   | 23.54   |
| Xverse-13B       | 12.55 | 23.49 | 5.20  | 17.69 | 5.54    | 7.46    | 10.19   |
| Baichuan-13B     | 12.38 | 22.46 | 5.21  | 17.59 | 5.42    | 7.37    | 10.03   |
| Baichuan2-13B    | 12.14 | 21.85 | 5.05  | 17.15 | 5.35    | 7.24    | 9.81    |
| Qwen-14B         | 11.90 | 22.43 | 4.89  | **16.94** | 5.24    | 7.03    | 9.67    |
| InternLM-20B     | 12.34 | 22.06 | 5.75  | 17.45 | 5.73    | 7.78    | 10.34   |
| Aquila2-34B      | 14.62 | 29.09 | 5.72  | 21.78 | 5.83    | 8.45    | 11.73   |
| Skywork-13B-Base | **11.58** | **21.84** | **4.76**  | 17.28 | **4.92**    | **6.82**    | **9.42**    | 


### Loss evaluation data and evaluation script
We have also open-sourced the data and evaluation scripts. You can reproduce our results by running the following command.

```
bash bash_scripts/skywork_eval_loss.sh
```

If you need to calculate the normalized loss for Model A and Skywork model, you can follow these steps:

1. Run the above script for Model A and Skywork model separately. The results will be stored in the result.txt file in their respective directories.
2. In the result.txt file, you will find two values for each model: the first value represents the loss, and the second value represents the number of document tokens.
3. Let's denote the loss and token numbers for Model A as loss_a and token_a, and for Skywork model as loss_s and token_s.
4. To calculate the normalized loss for Model A (loss_a_norm), loss_a_norm = loss_a * token_a / token_s
5. By comparing the normalized loss (loss_a_norm) of Model A with the loss (loss_s) of the Skywork model, we can evaluate the effectiveness of both models.
6. The same approach can be extended to multiple models.

### FAQ in Evaluation

**Q1**: Why should all models have the same document length instead of having the same number of tokens after tokenization?

**A1**: Essentially, domain perplexity measures the probability of different models generating high-quality documents, with higher probability indicating better model performance. Therefore, we need to ensure that all models see the same document. Additionally, since different models use different tokenizers, there can be a significant difference in the number of tokens after tokenization. For example, Llama will split Chinese characters into three Unicode encodings. If we compare the number of tokens after tokenization, the document length seen by Llama will be shorter compared to other models. However, we know that the token loss is lower in the beginning of the document and higher towards the end. Therefore, comparing based on the number of tokens after tokenization would be unfair to models like Llama, which have finer tokenization.

**Q2**: Why do we truncate the text to a length of max_position_embedding divided by 3?

**A2**: As mentioned in the answer to question 1, Llama model generally splits Chinese characters into three characters. To ensure that the maximum length of a document input to the model does not exceed the limit of 4096, we set the maximum document length to 4096/3 = 1228. In our comparison models, Llama has the finest tokenization for Chinese. Therefore, as long as the document length does not exceed the tokenization length of Llama, it will fit in other models as well.

**Q3**: Is it unfair to use a uniform length of 4096 for different models?

**A3**: As explained above, the calculated document length is 1228 Chinese characters. Taking Qwen as an example, the training length is 2K, which can be expanded to 8K during inference. Additionally, the compression ratio of bilingual models is generally 2-3 times. Therefore, 1228 Chinese characters usually only amount to 500-1000 tokens, far from the maximum length limit of 2K or even 4K.


**Q4**: Why is the Average Ppl inconsistent with the average Ppl of each domain?

**A4**: We calculate Average Ppl by averaging the losses of all documents and then converting it to Ppl using an exponential function. This is done to avoid having some domains with excessively high Ppl, which would negatively impact the overall results. The idea behind Average Ppl is to consider all documents as a cohesive collection, representing the overall Ppl of the document.

## Benchmark Results
We evaluated Skywork-13B-Base on several popular benchmarks, including C-Eval, MMLU, CMMLU, and GSM8K. Following the previous evaluation process, we tested the 5-shot results of C-Eval, MMLU, and CMMLU, and the 8-shot results of GSM8K. It can be seen that the Skywork-13B-Base model is among the top models in the Chinese open source model community, performing at an optimal level with the same parameter scale.

| Model            | C-Eval  | CMMLU | MMLU | GSM8K |  
|-------------------------|:-----:|:---------------:|:----------:|:-------:|
| LLaMA-1-13B-Base            | 35.5  | 31.2            | 46.9       | 17.8   | 
| Open-LLaMA-13B | 27.1  | 26.7         | 42.7       | 12.4   |
| LLaMA-2-13B-Base             | 36.5  | 36.6            | 54.8      | 28.7   | 
| InternLM-20B  | 58.8  |     -        |  62.0      | 52.6   | 
| Qwen-14B-Base | 72.1  |  71.0           | 66.3       | 61.3   |
| Aquila2-34B-Base | 63.1  |  71.4           | 64.2       | 58.4   |
| XVERSE-13B-Base              | 54.7  | -           | 55.1       | -   | 
| Baichuan-13B-Base | 52.4  | 55.3            | 51.6      | 26.6   |
| Baichuan-2-13B-Base | 58.1  | 62.0            | 59.2       | 52.3  |
| Skywork-13B-Base (ours)   | 60.6 | 61.8 | 62.1    | 55.8 | 

## Detailed Benchmark Results
We provide detailed results of the Skywork-13B-Base model on C-EVAL, CMMLU, and MMLU.

| Benchmark | **STEM** | **Humanities** | **Social Science** | **Other** | **China Specific** | **Hard** | **Average** | 
|:-----:|:---------:|:--------:|:-------------:|:--------:|:--------:|:--------:|:--------:|
| **C-EVAL** |   51.2   | 67.8    | 74.6        |  57.5   | - | 39.4   |  60.6   |
| **CMMLU**   |   49.5   | 69.3    | 65.9        |  63.3   | 64.2 | -   |  61.8   |
| **MMLU**   |   51.6   | 58.0    | 72.5       |  68.8   | - | -   |  62.1   |

## Reproduction

For your reproduction of the model performance on benchmark datasets, we provide scripts for you to reproduce the results. Check [eval/EVALUATION.md](eval/EVALUATION.md) for more information. Note that the reproduction may lead to slight differences from our reported results.

## Evaluation of Skywork-13B-Math
Skywork-13B-Math has further enhanced mathematical capabilities compared to the Base model. We conducted evaluations on mainstream mathematical related benchmarks, GSM8K, MATH, and CMATH. The results show that in the 13B scale model, our model ranked 1st in the GSM8K and CMATH benchmarks, and is also at the forefront in the MATH benchmark.

| Model            | GSM8K  | MATH | CMATH | 
|-------------------------|:-----:|:---------------:|:----------:|
| LLaMA-1-13B-Base            | 17.80  | 3.90            | -      |
| LLaMA-2-13B-Base             | 28.70  | 3.90            | -     |
| Baichuan-13B-Base | 26.76  | 4.84            | 51.33      |
| Baichuan-2-13B-Base | 52.77  | 10.08          | -       | 
| WizardMath-13B | 63.90  | 14.00            | 50.83      | 
| GAIRMATH-Abel-13B | 66.41  | 17.34            | -       | 
| MetaMath-13B | 72.30  | 22.40            | -       | 
| Skywork-13B-Math (ours)   | **72.33** | 16.98 | **77.27**    | 

# Quickstart
We have open-sourced the model parameters, configuration files, tokenizer, and more on Hugging Face and ModelScope.
## Requirements
- Python 3.8 and above
- Pytorch 2.0 and above 
- CUDA 11.4 and above are recommended.

Skywork-13B-Base model, Skywork-13B-Chat model, and Skywork-13B-Math model run the following script for Python dependency installation:
```shell
pip install -r requirements.txt 
```
## Demonstration of Hugging Face Model Inference


### Chat Model Inference
```python
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

    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, config=config, device_map="balanced_low_0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        
    return model, tokenizer


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


if __name__ == '__main__':
    tokenizer_path='skywork/skywork-13b-chat'
    checkpoint_path = 'skywork/skywork-13b-chat'
    model, tokenizer = load(tokenizer_path, checkpoint_path)

    doc = "how are you"
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

```

### Math Model Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer_path = ""
checkpoint_path = ""

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path, use_fast=False, trust_remote_code=True, padding_side='left')

model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path, device_map="auto", trust_remote_code=True).eval()
tokenizer.add_tokens(["[USER]", "[BOT]", "[SEP]"])

def special_encode(input, tokenizer):
    raw_str = "[USER]%s[SEP][BOT]" % input.strip().replace("\r", "")
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id
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

if __name__ == '__main__':
    text="Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    text_token_ids = torch.tensor(special_encode(
        text, tokenizer)).to(model.device).reshape(1, -1)
    response = model.generate(text_token_ids, do_sample=False, max_length=512)
    response_text = tokenizer.decode(response.cpu()[0], skip_special_tokens=True)
    response_text = extract_res(response_text)
    print(response_text)    
    """Skywork-13B-Math Response:
    First, we need to find out how many eggs Janet has left after eating for breakfast and baking for her friends. \n\nShe has 16 eggs per day, eats 3 for breakfast and uses 4 for baking. So, 16 - 3 - 4 = 9 eggs are left for selling at the farmers' market.\n\nSince she sells each egg for $2, she makes 9 * 2 = $<<9*2=18>>18 every day at the farmers' market.\n\nSo, the answer is $18.
    """
```


### CLI Demo 


```
python cli_demo.py \
    -m skywork-13b-chat-model-path 

```

<p align="center">
    <br>
    <img src="misc/chat_demo_1.gif" width="800" />
    <br>
<p>
<br>


<p align="center">
    <br>
    <img src="misc/chat_demo_2.gif" width="800" />
    <br>
<p>
<br>


<p align="center">
    <br>
    <img src="misc/chat_demo_3.gif" width="800" />
    <br>
<p>
<br>


# Quantization

## 8bit Quantization

Skywork utilizes the widely-used 8-bit quantization method called [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes). This method allows for quantizing performance with minimal loss and has been seamlessly integrated into the transformers library. Building upon BitsAndBytes, we offer two approaches for utilizing online quantization and offline 8-bit models.

To illustrate the usage of the int8 quantization model, we provide an example. Before you begin, please ensure that you have installed the BitsAndBytes library and the necessary dependencies. For detailed installation instructions, please refer to the [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) repository.

### Online Quantization

```python
model = AutoModelForCausalLM.from_pretrained("skywork-13B-Base", torch_dtype=torch.bfloat16,load_in_8bit=True, trust_remote_code=True).eval()
```

### Offline Quantization

```python
model = AutoModelForCausalLM.from_pretrained("skywork-13B-Base-8bits", device_map="auto", torch_dtype=torch.bfloat16,trust_remote_code=True).eval()
```



### Resutls

We have tested the quantitative model on benchmark evaluation datasets, and the results are as follows:

| Precision | C-Eval | MMLU  | CMMLU |
| --------- | ------ | ----- | ----- | 
| bf16      | 60.6  | 61.8 | 62.1 |
| 8bits     | 58.5  | 61.8 | 61.0 |


### GPU Mem in GB

| Precision | Skywork-13B |
| --------- | ----------- |
| bf16      | 25.91       |
| 8bits     | 13.57       |

# Fine-tuning
## Full-Parameter Fine-Tuning

Continuing pre-training with the Skywork-13B-Base model.
```bash
## preprocess continue pretraining data
## Because pre-training data is usually large, we use a script to process the training data separately.
python train/pt_data_preprocess.py \
    -t $MODEL_PATH \
    -i data/pt_train.jsonl \
    -o data_cache/pt_train_demo 

## launch training
export WANDB_API_KEY=YOUR_WANDB_KEY
export WANDB_ENTITY=skywork
export WANDB_PROJECT=skywork-13b-opensource

export MODEL_PATH=skywork-13b-models/skywork-13b-base
export DATA_CACHE_DIR=data_cache/pt_train_demo/pt_train
bash bash_scripts/skywork_13b_pt.sh
 
```

Conducting supervised fine-tuning with Skywork-13B-Base model.

```bash 
## preprocess data and launch training
export WANDB_API_KEY=YOUR_WANDB_KEY
export WANDB_ENTITY=skywork
export WANDB_PROJECT=skywork-13b-opensource

export SFT_DATA_DIR=data/sft_data
export DATA_CACHE_DIR=data_cache/sft_train_demo
bash bash_scripts/skywork_13b_sft.sh


```

## LoRA Fine-Tuning

Continuing LoRA pre-training with the Skywork-13B-Base model with LoRA.
```bash 
## preprocess continue pretraining data
## Because pre-training data is usually large, we use a script to process the training data separately.
python train/pt_data_preprocess.py \
    -t $MODEL_PATH \
    -i data/pt_train.jsonl \
    -o data_cache/pt_train_demo 


export WANDB_API_KEY=YOUR_WANDB_KEY
export WANDB_ENTITY=skywork
export WANDB_PROJECT=skywork-13b-opensource

export MODEL_PATH=/data/shared/public/liang.zhao/skywork-13b-models/skywork-13b-base
export DATA_CACHE_DIR=data_cache/pt_train_demo/pt_train
bash bash_scripts/skywork_13b_pt_lora.sh
 
```

Conducting supervised fine-tuning with Skywork-13B-Base model with LoRA.

```bash 

export WANDB_API_KEY=YOUR_WANDB_KEY
export WANDB_ENTITY=skywork
export WANDB_PROJECT=skywork-13b-opensource

export SFT_DATA_DIR=/data/user/liang.zhao/Skywork/data/sft_data
export DATA_CACHE_DIR=data_cache/sft_train_demo
bash bash_scripts/skywork_13b_sft_lora.sh
 
```

# Community and Ecosystem
## Huawei Ascend
### MindSpore Framework
[MindFormers]( https://gitee.com/mindspore/mindformers)is a full-process development suite based on the MindSpore framework that supports large model training, fine-tuning, evaluation, inference, and deployment. The [Skywork-13B]( https://gitee.com/mindspore/mindformers/tree/dev/research/skywork) model is integrated into this suite, supporting users to fine-tune and deploy models based on Ascend AI hardware capabilities. The specific usage can be seen in our[README]( https://gitee.com/mindspore/mindformers/tree/dev/research/skywork/skywork.md) on the MindSpore platform.

### Large Model Experience Platform
The [MindSpore Large Model Platform](​https://xihe.mindspore.cn) is based on the MindSpore AI framework, MindFormers large model development suite, and Ascend hardware capabilities, opening the [Skywork-13B](https://xihe.mindspore.cn/modelzoo/skywork_13b) large model capabilities to the public. Everyone is welcome to use it.

# Declaration and License Agreement


## Declaration

We hereby declare that the Skywork model should not be used for any activities that pose a threat to national or societal security or engage in unlawful actions. Additionally, we request users not to deploy the Skywork model for internet services without appropriate security reviews and records. We hope that all users will adhere to this principle to ensure that technological advancements occur in a regulated and lawful environment.

We have done our utmost to ensure the compliance of the data used during the model's training process. However, despite our extensive efforts, due to the complexity of the model and data, there may still be unpredictable risks and issues. Therefore, if any problems arise as a result of using the Skywork open-source model, including but not limited to data security issues, public opinion risks, or any risks and problems arising from the model being misled, abused, disseminated, or improperly utilized, we will not assume any responsibility.

## License Agreement

The community usage of Skywork model requires [Skywork Community License](https://github.com/SkyworkAI/Skywork/blob/main/Skywork%20Community%20License.pdf). The Skywork model supports commercial use. If you plan to use the Skywork model or its derivatives for commercial purposes, you must abide by terms and conditions within [Skywork Community License](https://github.com/SkyworkAI/Skywork/blob/main/Skywork%20Community%20License.pdf).


# Contact Us and Citation
If you find our work helpful, please feel free to cite our paper~
```
@misc{wei2023skywork,
      title={Skywork: A More Open Bilingual Foundation Model}, 
      author={Tianwen Wei and Liang Zhao and Lichang Zhang and Bo Zhu and Lijie Wang and Haihua Yang and Biye Li and Cheng Cheng and Weiwei Lü and Rui Hu and Chenxia Li and Liu Yang and Xilin Luo and Xuejie Wu and Lunan Liu and Wenjun Cheng and Peng Cheng and Jianhao Zhang and Xiaoyu Zhang and Lei Lin and Xiaokun Wang and Yutuan Ma and Chuanhai Dong and Yanqi Sun and Yifu Chen and Yongyi Peng and Xiaojuan Liang and Shuicheng Yan and Han Fang and Yahui Zhou},
      year={2023},
      eprint={2310.19341},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


```
@article{skyworkmath,
  title={SkyMath: Technical Report},
  author={Liu Yang, Haihua Yang, Wenjun Cheng, Lei Lin, Chenxia Li, Yifu Chen, Lunan Liu, Jianfei Pan, Tianwen Wei, Biye Li, Liang Zhao, Lijie Wang, Bo Zhu, Guoliang Li, Xuejie Wu, Xilin Luo, Rui Hu},
  journal={arXiv preprint arXiv: 2310.16713},
  url={https://arxiv.org/abs/2310.16713},
  year={2023}
}
```


```
@article{Skywork_Multi-Modal_Group_Empirical_Study_Towards_2023,
    author = {Skywork Multi-Modal Group},
    month = sep,
    title = {{Empirical Study Towards Building An Effective Multi-Modal Large Language Model}},
    year = {2023}
}

```
