import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os

model_path = "qwen3_int8.onnx"
tokenizer_path = "./Qwen3-1.7B"

sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# 动态获取模型输入名
input_names = [inp.name for inp in sess.get_inputs()]
print(f"Model inputs: {input_names}")

# ================= TODO 6: 实现自回归生成循环 =================
FIXED_SEQ_LEN = 32  # 必须与导出时的 mask 尺寸一致

def generate(prompt, max_tokens=50):
    # 1. 预处理 Prompt
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    
    print(f"Qwen: ", end="", flush=True)
    
    for _ in range(max_tokens):
        # 使用滑动窗口保持固定长度
        if input_ids.shape[1] > FIXED_SEQ_LEN:
            input_window = input_ids[:, -FIXED_SEQ_LEN:]
            actual_len = FIXED_SEQ_LEN
        else:
            # 右填充 pad_token（因果LM标准做法）
            actual_len = input_ids.shape[1]
            pad_len = FIXED_SEQ_LEN - actual_len
            padding = np.full((1, pad_len), tokenizer.pad_token_id, dtype=np.int64)
            input_window = np.concatenate([input_ids, padding], axis=1)
        
        # 1. 构造推理输入字典
        ort_inputs = {"input_ids": input_window}
        if "attention_mask" in input_names:
            # 右填充时 attention_mask 前面是1，后面是0
            attention_mask = np.zeros_like(input_window, dtype=np.int64)
            attention_mask[0, :actual_len] = 1
            ort_inputs["attention_mask"] = attention_mask
        
        # 2. 执行推理
        outputs = sess.run(None, ort_inputs)
        logits = outputs[0]
        
        # 3. 获取下一个 token（取实际序列最后一个位置，而非固定-1）
        next_token = int(np.argmax(logits[0, actual_len - 1, :]))
        
        # 4. 结束条件判断
        if next_token == tokenizer.eos_token_id:
            break
        
        # 5. 打印当前生成的字
        word = tokenizer.decode([next_token])
        print(word, end="", flush=True)
        
        # 6. 更新 input_ids（完整序列，用于下一次滑动窗口）
        input_ids = np.append(input_ids, [[next_token]], axis=1)
        
    print("\n")

while True:
    q = input("\nUser: ")
    if q == "exit": break
    generate(q)