from transformers import AutoTokenizer, AutoModelForCausalLM
from specSampling import specSampling
from regrSampling import regrSampling
import torch

p_model_path = "./models/pythia-2.8b"
q_model_path = "./models/pythia-70m"
print(f"正在加载模型...")

tokenizer = AutoTokenizer.from_pretrained(q_model_path)

p_model = AutoModelForCausalLM.from_pretrained(
    p_model_path, device_map="auto", dtype=torch.float16
)
q_model = AutoModelForCausalLM.from_pretrained(
    q_model_path, device_map="auto", dtype=torch.float16
)
device = p_model.device
print("模型加载完成！")
print(f"模型当前运行在: {device}")

text = "Computer science is"
inputs = tokenizer(text, return_tensors="pt")
prefix = inputs["input_ids"].to(device)

result = regrSampling(
    x=prefix,
    model=q_model,
    maxLen=200,
    temperature=0.85,
    top_p=0.9
)
output_text = tokenizer.decode(result[0], skip_special_tokens=True)
print(output_text)

result = specSampling(
    prefix=prefix,
    q_model=q_model,
    p_model=p_model,
    maxLen=200,
    gamma=4,
    temperature=0.85,
    top_p=0.9
)
output_text = tokenizer.decode(result[0], skip_special_tokens=True)
print(output_text)

