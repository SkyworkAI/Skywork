## 评测复现

- CEVAL

```Shell
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
mkdir data/ceval
mv ceval-exam.zip data/ceval
cd data/ceval; unzip ceval-exam.zip
cd ../../

# Skywork-13B-Base
python evaluate_ceval.py -d data/ceval/

```

- MMLU

```Shell
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
cd ../../

# Skywork-13B-Base
python evaluate_mmlu.py -d data/mmlu/data/
```

- CMMLU

```Shell
wget https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip
mkdir data/cmmlu
mv cmmlu_v1_0_1.zip data/cmmlu
cd data/cmmlu; unzip cmmlu_v1_0_1.zip
cd ../../

# Skywork-13B-Base
python evaluate_cmmlu.py -d data/cmmlu/
```
