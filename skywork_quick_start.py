
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-13B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Skywork/Skywork-13B-Chat", device_map="auto", trust_remote_code=True).eval()

inputs = tokenizer('陕西的省会是西安', return_tensors='pt').to(model.device)
response = model.generate(inputs.input_ids, max_length=128)
print(tokenizer.decode(response.cpu()[0], skip_special_tokens=True))


inputs = tokenizer('陕西的省会是西安，甘肃的省会是兰州，河南的省会是郑州', return_tensors='pt').to(model.device)
response = model.generate(inputs.input_ids, max_length=128)
print(tokenizer.decode(response.cpu()[0], skip_special_tokens=True))
