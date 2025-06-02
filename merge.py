from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 配置路径
base_model_name = "/root/autodl-tmp/models/meta-llama/Llama-2-7b-chat-hf"  # 替换为你的基础模型名称或路径
adapter_config_path = "/root/autodl-tmp/models/llama2-baseline/saferlhf/negative-0.65/adapters/original"  # 替换为你的 adapter 配置文件路径
adapter_weights_path = "/root/autodl-tmp/models/llama2-baseline/saferlhf/negative-0.65/adapters/original"  # 替换为你的 adapter 权重路径
merged_model_save_path = "/root/autodl-tmp/models/llama2-baseline/saferlhf/negative-0.65/merge/original"  # 替换为你希望保存合并后模型的路径

# 1. 加载基础模型
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# 2. 加载 PEFT 配置
peft_config = PeftConfig.from_pretrained(adapter_config_path)

# 3. 加载带有 adapter 的 PEFT 模型
peft_model = PeftModel.from_pretrained(model, adapter_weights_path)

# 4. 合并 adapter 权重到基础模型
peft_model.merge_and_unload()

# 5. 验证合并是否成功（可选）
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# 6. 保存合并后的模型
model.save_pretrained(merged_model_save_path)
tokenizer.save_pretrained(merged_model_save_path)

print(f"Merged model saved to {merged_model_save_path}")