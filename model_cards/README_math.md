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
**Skywork-13B-Math**模型经过专门的数学能力强化训练。在13B规模的模型中，Skywork-13B-Math模型在GSM8K评测上得分第一，同时在MATH数据集以及CMATH上也表现优异，处于13B模型顶尖水平。

**Skywork-13B-Math**: Skywork-13B-Math model has undergone specialized training to enhance its mathematical abilities. In the 13B-scale model, the Skywork-13B-Math model ranked first in the GSM8K evaluation, and it also performed exceptionally well on the MATH dataset and CMATH, placing it among the top-level 13B models.


如果您希望了解更多的信息，如训练方案，评估方法，请参考我们的[技术报告](https://arxiv.org/skywork-tech-report)和[Skywork-Math](https://arxiv.org/skywork-tech-report)论文。

If you are interested in more training and evaluation details, please refer to our [technical report](https://arxiv.org/skywork-tech-report) and [Skywork-Math]((https://arxiv.org/skywork-tech-report)) paper.


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
## Huggingface模型测试（Demostration）


### Math 模型推理（Math Model Inferecen）
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
if __name__ == '__main__':
    text = "小王要将150千克含药量20%的农药稀释成含药量5%的药水．需要加水多少千克？"
    text_token_ids = torch.tensor(special_encode(
        text, tokenizer)).to(model.device).reshape(1, -1)
    response = model.generate(text_token_ids, do_sample=False, max_length=512)
    response_text = tokenizer.decode(response.cpu()[0], skip_special_tokens=True).split(
        "[BOT]")[-1].split("[SEP]")[0].strip()
    print(response_text)   
    """输出结果：
    首先，我们需要计算出150千克含药量20%的农药中含有多少千克的药。\n\n150千克 * 20% = 30千克\n\n然后，我们需要计算出要得到含药量5%的药水，需要多少千克的药水。\n\n30千克 / 5% = 600千克\n\n最后，我们需要计算出需要加多少千克的水。\n\n600千克 - 150千克 = 450千克\n\n所以答案是，小王需要加450千克的水。
    """ 
```

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
if __name__ == '__main__':
    text="Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    text_token_ids = torch.tensor(special_encode(
        text, tokenizer)).to(model.device).reshape(1, -1)
    response = model.generate(text_token_ids, do_sample=False, max_length=512)
    response_text = tokenizer.decode(response.cpu()[0], skip_special_tokens=True).split(
        "[BOT]")[-1].split("[SEP]")[0].strip()
    print(response_text)    
    """Skywork-13B-Math Response:
    First, we need to find out how many eggs Janet has left after eating for breakfast and baking for her friends. \n\nShe has 16 eggs per day, eats 3 for breakfast and uses 4 for baking. So, 16 - 3 - 4 = 9 eggs are left for selling at the farmers' market.\n\nSince she sells each egg for $2, she makes 9 * 2 = $<<9*2=18>>18 every day at the farmers' market.\n\nSo, the answer is $18.
    """
```

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
