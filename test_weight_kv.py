from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = r'E:\llama-3.2-1b\Llama-3.2-1B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

output_file = "model_layer_kvcache_size_2_13B.txt"

input_text = "Hello " * 500
input_ids = tokenizer(input_text, return_tensors="pt", max_length=2000, truncation=True).input_ids.to(device)

print(f"Input shape: {input_ids.shape}")

input_length = input_ids.shape[1]

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // num_heads

print(f"Expected number of layers: {num_layers}")

weight_sizes = [0] * num_layers
kv_cache_sizes = [0] * num_layers
kv_cache_per_token_sizes = [0] * num_layers

total_weight_size = 0

element_size = torch.tensor(0, dtype=torch.float16, device=device).element_size()

for layer_idx in range(num_layers):
    layer_weight_size = 0
    for name, param in model.named_parameters():
        if f"model.layers.{layer_idx}." in name:
            layer_weight_size += param.numel() * param.element_size()
    
    weight_sizes[layer_idx] = layer_weight_size
    total_weight_size += layer_weight_size
    
    tensor_size = input_length * head_dim
    kv_cache_sizes[layer_idx] = 2 * num_heads * tensor_size * element_size
    kv_cache_per_token_sizes[layer_idx] = 2 * num_heads * head_dim * element_size

with torch.no_grad():
    output = model(input_ids)

print(f"Weight sizes length: {len(weight_sizes)}, KV Cache sizes length: {len(kv_cache_sizes)}, Expected layers: {num_layers}")

with open(output_file, "w") as f:
    f.write("Prefill Phase Analysis:\n")
    for i in range(num_layers):
        f.write(f"Layer {i}: Weight Size: {weight_sizes[i] / (1024 ** 3):.2f} GB, KV Cache Size: {kv_cache_sizes[i] / (1024 ** 3):.2f} GB, KV Cache Per Token: {kv_cache_per_token_sizes[i]} bytes\n")
    
    f.write(f"\nTotal Model Weight Size: {total_weight_size / (1024 ** 3):.2f} GB\n")

print(f"End！ {output_file}")

