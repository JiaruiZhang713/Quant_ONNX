import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# ================= TODO 1: 实现掩码补丁 =================
# 提示：Qwen3 原生代码中的 mask 生成逻辑包含 ONNX 不支持的算子。
# 你需要编写一个函数，根据输入的 input_ids 形状，生成一个上三角掩码矩阵。
# 要求：
# 1. 能够从 kwargs 中尝试获取 input_shape (batch, seq_len)
# 2. 生成一个全为负无穷(float.min)的矩阵，仅保留上三角(triu)
# 3. 返回形状必须是 (batch, 1, seq_len, seq_len)
def mask_patch(*args, **kwargs):
    # 1. 解析参数
    if "input_shape" in kwargs:
        bsz, seq_len = kwargs["input_shape"]
    else:
        bsz, seq_len = 1, 32
    
    dtype = kwargs.get("dtype", torch.float32)
    device = kwargs.get("device", torch.device("cpu"))

    # 2. 生成上三角因果掩码
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)
    
    return mask

# 应用补丁
transformers.masking_utils.create_causal_mask = mask_patch
print(">>> [Patch Applied] 已应用掩码补丁")


# ================= TODO 2: 实现模型包装器 (Wrapper) =================
# 提示：ONNX 导出时不支持 transformers 输出的 DynamicCache 对象。
# 你需要封装原模型，强制关闭缓存，并只返回 logits。
class Qwen3ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return outputs.logits


# ================= 主程序 =================
model_path = "./Qwen3-1.7B"
output_file = "qwen3_fp32.onnx"

print(f"--- Loading Model ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float32, 
        device_map="cpu", 
        trust_remote_code=True,
        attn_implementation="eager"
    )
    base_model.eval()
except Exception as e:
    print(f"Error: {e}")
    exit(1)

model_wrapper = Qwen3ONNXWrapper(base_model)

# 构造虚拟输入
dummy_input_ids = torch.ones((1, 32), dtype=torch.long)
dummy_attention_mask = torch.ones((1, 32), dtype=torch.long)

print(f"--- Exporting to {output_file} ---")

# ================= TODO 3: 配置导出参数 =================
# 提示：请查阅 torch.onnx.export 文档
with torch.no_grad():
    torch.onnx.export(
        model_wrapper,
        (dummy_input_ids, dummy_attention_mask),
        output_file,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"}
        },
        
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )

print(f"✅ Export Success!")