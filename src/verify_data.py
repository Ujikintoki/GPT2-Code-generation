import os
from datasets import load_from_disk
from transformers import AutoTokenizer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "processed", "debug_sample")

print(f"尝试加载本地数据，路径: {data_path}")

# 2. 加载数据和分词器
dataset = load_from_disk(data_path)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 3. 验证数据维度 (应输出 256)
sample_input = dataset[0]["input_ids"]
print(f"张量长度验证: {len(sample_input)}") 

# 4. 验证数据重组逻辑 (把 ID 反向解码成人类代码)
print("\n--- 解码第一个 Chunk (0-256) ---")
print(tokenizer.decode(sample_input))

# 5. 验证滑动窗口逻辑 
print("\n--- 解码第二个 Chunk (10-266) ---")
print(tokenizer.decode(dataset[1]["input_ids"]))