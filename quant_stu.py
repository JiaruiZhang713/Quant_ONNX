import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from transformers import AutoTokenizer
import numpy as np
import os

# ================= TODO 4: 实现校准数据读取器 =================
class SmartCalibrationDataReader(CalibrationDataReader):
    def __init__(self, tokenizer, model_path, num_samples=1000):
        self.tokenizer = tokenizer
        # 自动获取模型输入名 (防止 input name mismatch)
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_names = [inp.name for inp in session.get_inputs()]
        
        # 使用聊天格式的校准数据（更匹配实际使用场景）
        print(f"Loading calibration dataset ({num_samples} samples)...")
        from datasets import load_dataset
        
        # 加载 wikitext 作为基础文本
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        raw_texts = [t for t in dataset["text"] if len(t.strip()) > 30][:num_samples]
        
        # 转换为聊天格式，与 chatbot 使用的格式一致
        chat_texts = []
        for text in raw_texts:
            formatted = f"<|im_start|>user\n{text[:100]}<|im_end|>\n<|im_start|>assistant\n"
            chat_texts.append(formatted)
        
        print(f"Loaded {len(chat_texts)} calibration samples (chat format)")
        self.data = iter(chat_texts)

    def get_next(self):
        text = next(self.data, None)
        if text is None: return None
        
        # 添加 padding 到固定长度，与导出时的 mask 尺寸匹配
        encoded = self.tokenizer(
            text, 
            return_tensors="np",
            padding="max_length",
            max_length=32,
            truncation=True
        )
        result = {}
        if "input_ids" in self.input_names:
            result["input_ids"] = encoded["input_ids"].astype(np.int64)
        if "attention_mask" in self.input_names:
            result["attention_mask"] = encoded["attention_mask"].astype(np.int64)
        return result

# 主程序
model_fp32 = "qwen3_fp32.onnx"
model_int8 = "qwen3_int8.onnx"

if not os.path.exists(model_fp32):
    print("未找到 FP32 模型，请先完成任务一。")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained("./Qwen3-1.7B", trust_remote_code=True)
dr = SmartCalibrationDataReader(tokenizer, model_fp32)

print("--- Starting Quantization ---")

# ================= TODO 5: 执行静态量化 =================
# 提示：由于模型大于 2GB，直接量化会报错 Protobuf parsing failed。
# 你需要设置哪个参数来启用外部数据存储？
quantize_static(
    model_input=model_fp32,
    model_output=model_int8,
    calibration_data_reader=dr,
    quant_format=onnxruntime.quantization.QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    use_external_data_format=True,
)

print(f"✅ Quantization Complete!")