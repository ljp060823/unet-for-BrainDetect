from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/data/huatuo_qwen3.5_9b/qwen3.5_9b" 

# 直接加载合并后的模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 测试对话
prompt = "怀孕后嘴巴很淡怎么办"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))