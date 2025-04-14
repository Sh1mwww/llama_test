import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_path = 'D:\\huggingface_model\\Llama-3.2-1B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

input_txt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'

input_ids = tokenizer(input_txt, return_tensors="pt").input_ids

output = model.generate(
    input_ids,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=100,
    output_scores=False,
    output_attentions=False,
    output_hidden_states=False,
    return_dict_in_generate=True
)

# 最后一个 token 的 past_key_values
past_key_values = output.past_key_values

kv_summary = []
total_size_bytes = 0
bytes_per_element = 2  # float16 = 2 bytes

for layer_idx, (key, value) in enumerate(past_key_values):
    key_shape = list(key.shape)
    value_shape = list(value.shape)

    key_numel = key.numel()
    value_numel = value.numel()

    layer_total_bytes = (key_numel + value_numel) * bytes_per_element
    total_size_bytes += layer_total_bytes

    kv_summary.append({
        "layer": layer_idx,
        "key_shape": key_shape,
        "value_shape": value_shape,
        "key_numel": key_numel,
        "value_numel": value_numel,
        "layer_total_bytes": layer_total_bytes
    })

# 添加总大小统计
summary = {
    "kv_cache_per_layer": kv_summary,
    "total_kv_cache_bytes": total_size_bytes,
    "total_kv_cache_MB": round(total_size_bytes / (1024 * 1024), 2)
}

# 保存结果
output_file = "kv_cache_summary.json"
with open(output_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"KV cache summary saved to {output_file}")
