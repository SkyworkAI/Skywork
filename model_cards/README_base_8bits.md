<!-- <div align="center">
<h1>
  ✨Skywork
</h1>
</div> -->
<div align="center"><img src="misc/skywork_logo.jpeg" width="550"/></div>

<p align="center">
🤗 <a href="https://huggingface.co/Skywork" target="_blank">Hugging Face</a> • 🤖 <a href="https://modelscope.cn/organization/Skywork" target="_blank">ModelScope</a> • 💬 <a href="https://github.com/SkyworkAI/Skywork/blob/main/misc/wechat.png?raw=true" target="_blank">WeChat</a>• 📜<a href="https://arxiv.org/" target="_blank">Tech Report</a>• 🧮<a href="https://arxiv.org/" target="_blank">Skymath Paper</a>
</p>


<div align="center">


[🎉天工在线对话平台已正式向公众开放](https://sso.tiangong.cn/?redirect=https://model-platform.tiangong.cn/overview&client_id=200005)

</div>



<div align="center">


[![GitHub Stars](https://img.shields.io/github/stars/SkyworkAI/Skywork)](https://github.com/SkyworkAI/Skywork/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/SkyworkAI/Skywork)](https://github.com/SkyworkAI/Skywork/fork)
</div>



# 模型介绍（Introduction）
**Skywork-13B-Base**模型在高质量清洗过滤的3.2万亿个多语言（主要是中文和英文）和代码数据上进行预训练，它在多种评测和各种基准测试上都展现了同等规模模型的最佳效果。

**Skywork-13B-Base**: The model was trained on a high-quality cleaned dataset consisting of 3.2 trillion multilingual data (mainly Chinese and English) and code. It has demonstrated the best performance among models of similar scale in various evaluations and benchmark tests.

如果您希望了解更多的信息，如训练方案，评估方法，请参考我们的[技术报告](https://arxiv.org/skywork-tech-report)和[Skywork-Math](https://arxiv.org/skywork-tech-report)论文。

If you are interested in more training and evaluation details, please refer to our [technical report](https://arxiv.org/skywork-tech-report) and [Skywork-Math]((https://arxiv.org/skywork-tech-report)) paper.


## 训练数据（Training Data）
我们精心搭建了数据清洗流程对文本中的低质量数据、有害信息、敏感信息进行清洗过滤。我们的Skywork-13B-Base模型是在清洗后的3.2TB高质量中、英、代码数据上进行训练，其中英文占比52.2%，中文占比39.6%，代码占比8%，在兼顾中文和英文上的表现的同时，代码能力也能有保证。

We have developed a data cleaning pipeline with great care to effectively clean and filter low-quality data and eliminate harmful information from text data. Our Skywork-13B-Base model is trained on a  dataset with 3.2TB tokens that consists of high-quality Chinese, English, and code data, all of which have been thoroughly cleaned. The English data comprises 52.2% of the dataset, the Chinese data accounts for 39.6%, and the code data makes up 8%. This comprehensive approach ensures optimal performance for both Chinese and English while also maintaining the ability to handle code.

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





## 模型结构（Model Structure）
与Llama-2-13B模型对比，天工Skywork-13B模型采用相对更加瘦长的网络结构，层数为52层，同时将FFN Dim和Hidden Dim缩小到12288和4608，从而保证模型参数量和原始Llama-2-13B模型相当。根据我们前期实验对比，相对瘦长的网络结构在大Batch Size训练下可以取得更好的泛化效果。Skywork-13B和Llama-2-13B模型的对比如下：

Compared to the Llama2-13B model, the Skywork-13B model adopts a relatively thinner and deeper network structure with 52 layers. At the same time, the FFN Dim and Hidden Dim are reduced to 12288 and 4608, respectively, to ensure that the model has a similar number of parameters as the original Llama-13B model. Based on our preliminary experimental results, a relatively thinner and deeper network structure can achieve better generalization performance under large batch size training. The detailed comparison between the Skywork-13B and Llama-2-13B models is as follows:

| Model Structure         | Llama2-13B | Skywork-13B | 
|----------------------|:----:|:-----------:|
| Vocab. Size  | 32,000 |    65,536     | 
| Hidden Dim.  | 5,120 |    4,608     | 
| FFN Dim.  | 13,696 |    12,288     |
| Head Dim. | 128 |    128     | 
| Num. Heads | 40 |    36     | 
| Num. Layers | 40 |    52     | 
| Seq. Len. | 4,096 |    4,096     | 
| Positional Embedding | RoPE | RoPE |


## 分词器（Tokenizer）
我们使用Byte-Pair Encoding（BPE）对数据进行分词，词表大小为65536，其中拉丁字符和子词为32000个，汉字和Unicode符号8000个，汉语词语25519个，剩下的17个为保留字。

We use Byte-Pair Encoding (BPE) to tokenize the data, with a vocabulary size of 65536. Among them, there are 32000 Latin characters and subwords, 8000 Chinese characters and Unicode symbols, 25519 Chinese words, and the remaining 17 are reserved words.


| Category                            | Size    |
|---------------------------------|--------|
| Latin based words & subwords                 | 32000  |
| Chinese characters & Unicode symbols               | 8000   |
| Chinese words                        | 25519  |
| Reserved symbols                       | 17     |
| **Total**                         | **65536** |


# 模型评估（Evaluation）
## 领域数据困惑度评估（Perplexity Evaluaiton）
语言模型训练的本质上是让预测下一个词更准确。基于这个认知，我们认为评估基础大模型一个重要的方式是评估在各大领域上语言模型生成文章的概率。在模型训练中预测下一个词的概率一般使用Cross Entropy损失函数，整体的损失函数为每个位置预测真实词损失的平均，则有：

$$loss = \sum^{n}_{i=1} log(p_i) / n = log( \prod_{i=1}^n p_i) / n$$

其中$n$是文档的长度，即token数，$p_i$是位置i上真实词的概率，我们知道文档中每一个位置上真实词的概率的联乘则为生成该文档的概率，如此我们就将loss和生成文章的概率联系在了一起。而不同模型因为使用的分词器不同，具有不同的token数，因此对损失函数乘以token数目$n$，这样就仅考虑生成文章的概率部分，不同模型也可以进行比较。我们将标准化后loss取指数转换成perplexity，使得模型的差异更加可读。为了阅读方便后续提到的loss和ppl为模型标准化后的loss和perplexity。

基于上述分析，我们对对多个领域筛选出2023年10月份新发布的几百到上千篇高质量文章，并人工进行了核对。保证所有的测试数据不在天工模型以及其他所有模型的训练集中，并且测试数据的来源也足够广泛，质量也高。我们可以选取当前最新的文章评测不同模型的ppl，模型很难作弊。
下图列出了不同开源模型，天工Skywork-13B-Base取得最优效果，证明了我们的Base模型的基础能力处于国内开源模型中文最强水平。

We have chosen several hundred to thousands of high-quality articles that were published in October 2023 across various fields. We have manually verified these articles to ensure their quality. It is important to note that none of the test data used in evaluating the Skywork model or any other models is included in their training set. Furthermore, the test data is diverse and of high quality, making it challenging for the models to gain an unfair advantage.

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

### 评测数据和评测脚本（Loss Evaluation）
我们将评测数据和评测脚本也进行了开源，下载github上的代码运行下面命令则可以复现我们的结果。

We have also open-sourced the data and evaluation scripts. You can reproduce our results by running the following command.

```
bash bash_scripts/skywork_eval_loss.sh
```

## Benchmark评估（Benchmark Results）
我们评估了各大权威评测benchmark上的结果作为参考，包括C-Eval，MMLU，CMMLU，GSM8K。遵循之前的评估流程，C-Eval、MMLU、CMMLU测试5-shot结果，GSM8K测试8-shot结果。可以看到Skywork-13B-Base模型在中文开源模型中处于前列，在同等参数规模下为最优水平。

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
| Skywork-13B-Base (ours)   | 59.5 | 61.6 | 61.6    | 55.8 | 

## Benchmark评估详细结果
我们给出**Skywork-13B-Base**模型在C-Eval，CMMLU，MMLU上模型的详细结果。

We provide detailed results of the Skywork-13B-Base model on C-EVAL, CMMLU, and MMLU.

| Benchmark | **STEM** | **Humanities** | **Social Science** | **Other** | **China Specific** | **Hard** | **Average** | 
|:-----:|:---------:|:--------:|:-------------:|:--------:|:--------:|:--------:|:--------:|
| **C-EVAL** |   51.5   | 65.1    | 73.9        |  55.1   | - | 39.9   |  59.5   |
| **CMMLU**   |   49.8   | 68.9    | 65.6        |  62.8   | 63.7 | -   |  61.6   |
| **MMLU**   |   50.6   | 57.8    | 71.9       |  68.3   | - | -   |  61.6   |


# 快速开始（Quickstart）
我们将模型参数、配置文件、tokenizer等在huggingface和modelscope上进行了开源。

We have open-sourced the model parameters, configuration files, tokenizer, and more on Huggingface and Modelscope.

## 依赖安装（Requirements）
- Python 3.8及以上版本
- Pytorch 2.0及以上版本
- CUDA建议使用11.4以上版本。

Skywork-13B-Base模型，Skywork-13B-Chat模型和Skywork-13B-Math模型运行下面的脚本进行Python依赖安装。

- Python 3.8 and above
- Pytorch 2.0 and above 
- CUDA 11.4 and above are recommended.

Skywork-13B-Base model, Skywork-13B-Chat model, and Skywork-13B-Math model run the following script for Python dependency installation:

```shell
pip install -r requirements.txt 
```
## Huggingface模型测试（Demonstration）


### Base 模型推理（Base Model Inference）

```python

>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from transformers.generation import GenerationConfig
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("SkyworkAI/Skywork-13B-Base", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("SkyworkAI/Skywork-13B-Base", device_map="auto", trust_remote_code=True).eval()

>>> inputs = tokenizer('陕西的省会是西安', return_tensors='pt').to(model.device)
>>> response = model.generate(inputs.input_ids, max_length=128)
>>> print(tokenizer.decode(response.cpu()[0], skip_special_tokens=True))
陕西的省会是西安，西安是我国著名的古都，在历史上有十三个朝代在此建都，所以西安又被称为“十三朝古都”。西安是我国著名的旅游城市，每年都有大量的游客来到西安旅游，西安的旅游资源非常丰富，有很多著名的旅游景点，比如秦始皇兵马俑、大雁塔、华清池、大唐芙蓉园、西安城墙、大明宫国家遗址公园、西安碑林博物馆、西安钟楼、西安鼓楼、西安半坡博物馆、西安大兴善寺、西安小雁塔


>>> inputs = tokenizer('陕西的省会是西安，甘肃的省会是兰州，河南的省会是郑州', return_tensors='pt').to(model.device)
>>> response = model.generate(inputs.input_ids, max_length=128)
>>> print(tokenizer.decode(response.cpu()[0], skip_special_tokens=True))
陕西的省会是西安，甘肃的省会是兰州，河南的省会是郑州，湖北的省会是武汉，湖南的省会是长沙，江西的省会是南昌，安徽的省会是合肥，江苏的省会是南京，浙江的省会是杭州，福建的省会是福州，广东的省会是广州，广西的省会是南宁，海南的省会是海口，四川的省会是成都，贵州的省会是贵阳，云南的省会是昆明，西藏的省会是拉萨，青海的省会是西宁，宁夏的省会是银川，新疆的省会是乌鲁木齐。


```

# 模型微调（Fine-tuning）
## 全量微调（Full-parameter Fine-tuning）
使用Skywork-13B-Base模型进行预训练微调
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
使用Skywork-13B-Base模型进行有监督微调（SFT, Supevise Fine-tuning）

```bash 
## preprocess data and launch training
export WANDB_API_KEY=YOUR_WANDB_KEY
export WANDB_ENTITY=skywork
export WANDB_PROJECT=skywork-13b-opensource

export SFT_DATA_DIR=data/sft_data
export DATA_CACHE_DIR=data_cache/sft_train_demo
bash bash_scripts/skywork_13b_sft.sh


```

## LoRA微调（PEFT）
使用Skywork-13B-Base模型以及LoRA进行预训练微调
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

export MODEL_PATH=skywork-13b-models/skywork-13b-base
export DATA_CACHE_DIR=data_cache/pt_train_demo/pt_train
bash bash_scripts/skywork_13b_pt_lora.sh
 
```

使用Skywork-13B-Base模型以及LoRA进行有监督微调（SFT, Supevise Fine-tuning）

```bash 


export WANDB_API_KEY=YOUR_WANDB_KEY
export WANDB_ENTITY=skywork
export WANDB_PROJECT=skywork-13b-opensource

export SFT_DATA_DIR=data/sft_data
export DATA_CACHE_DIR=data_cache/sft_train_demo
bash bash_scripts/skywork_13b_sft_lora.sh
 
```

# 量化部署（Quantization）

## 8bit量化（Int8 Quantization）

skywork 采用主流8bits量化方法：[BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)。该方法量化后性能基本无损，且已经集成到transformers库中，基于BitsAndBytes，我们提供在线量化和离线8bits模型两种方式。

以下我们提供示例说明如何使用int8量化模型，在开始使用之前，请先安装BitsAndBytes库并安装所需依赖包，具体安装方式见[BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)库。

### 在线量化（Online Quantization）

```python
model = AutoModelForCausalLM.from_pretrained("skywork-13B-Base", torch_dtype=torch.bfloat16,load_in_8bit=True, trust_remote_code=True).eval()
```

### 离线量化（Offline Quantization）

```python
model = AutoModelForCausalLM.from_pretrained("skywork-13B-Base-8bits", device_map="auto", torch_dtype=torch.bfloat16,trust_remote_code=True).eval()
```



### 量化效果（Evaluation）

我们对量化模型在基准评测数据集上做了测试，结果如下所示：

| Precision | C-Eval | MMLU  | CMMLU |
| --------- | ------ | ----- | ----- | 
| bf16      | 59.5  | 61.6 | 61.6 |
| 8bits     | 58.5  | 61.8 | 61.0 |

### 显存占用（GPU Mem in GB）

| Precision | Skywork-13B |
| --------- | ----------- |
| bf16      | 25.91       |
| 8bits     | 13.57       |



# 声明和协议（Declaration and License Aggrement）


## 声明（Declaration）

我们在此声明，不要利用Skywork模型进行任何危害国家社会安全或违法的活动。另外，我们也要求使用者不要将 Skywork 模型用于未经适当安全审查和备案的互联网服务。我们希望所有的使用者都能遵守这个原则，确保科技的发展能在规范和合法的环境下进行。

我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用skywork开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

We hereby declare that the Skywork model should not be used for any activities that pose a threat to national or societal security or engage in unlawful actions. Additionally, we request users not to deploy the Skywork model for internet services without appropriate security reviews and records. We hope that all users will adhere to this principle to ensure that technological advancements occur in a regulated and lawful environment.

We have done our utmost to ensure the compliance of the data used during the model's training process. However, despite our extensive efforts, due to the complexity of the model and data, there may still be unpredictable risks and issues. Therefore, if any problems arise as a result of using the Skywork open-source model, including but not limited to data security issues, public opinion risks, or any risks and problems arising from the model being misled, abused, disseminated, or improperly utilized, we will not assume any responsibility.

## 协议（License Aggrement）

社区使用Skywork模型需要遵循[《Skywork 模型社区许可协议》](https://github.com/SkyworkAI/Skywork/blob/main/Skywork%20模型社区许可协议.pdf)。Skywork模型支持商业用途，如果您计划将Skywork模型或其衍生品用于商业目的，无需再次申请， 但请您仔细阅读[《Skywork 模型社区许可协议》](https://github.com/SkyworkAI/Skywork/blob/main/Skywork%20模型社区许可协议.pdf)并严格遵守相关条款。 


The community usage of Skywork model requires [Skywork Community License](https://github.com/SkyworkAI/Skywork/blob/main/Skywork%20Community%20License.pdf). The Skywork model supports commercial use. If you plan to use the Skywork model or its derivatives for commercial purposes, you must abide by terms and conditions within [Skywork Community License](https://github.com/SkyworkAI/Skywork/blob/main/Skywork%20Community%20License.pdf).

  

[《Skywork 模型社区许可协议》》]:https://github.com/SkyworkAI/Skywork/blob/main/Skywork%20模型社区许可协议.pdf


[skywork-opensource@kunlun-inc.com]: mailto:skywork-opensource@kunlun-inc.com

# 引用和联系我们（Contact Us and Citation）
如果您觉得我们的工作对您有帮助，欢迎引用我们的论文~

If you find our work helpful, please feel free to cite our paper~
```
@article{skyworktechreport,
  title={},
  author={},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```

```
@article{skyworkmath,
  title={},
  author={},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```
