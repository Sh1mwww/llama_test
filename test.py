from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
model_path = 'D:\huggingface_model\Llama-3.2-1B'
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

# Prepare the input
input_txt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'

# Tokenize the input text
input_ids = tokenizer(input_txt, return_tensors="pt").input_ids

# Generate text
output = model.generate(
    input_ids,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
    output_scores=True,
    output_attentions=True,
    output_hidden_states=True,
    return_dict_in_generate=True  # This allows access to scores, attentions, etc.
)

# Decode the generated text
generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
print(f"Generated Text: {generated_text}")

# Output the scores (logits before softmax)
if 'scores' in output:
    print("Scores:", output.scores)

# Output the attentions if available
if 'attentions' in output:
    print("Attentions:", output.attentions)

# Output the hidden states if available
if 'hidden_states' in output:
    print("Hidden States:", output.hidden_states)

# If you want to access past key values (model cache) for faster generation:
if 'past_key_values' in output:
    print("Past Key Values:", output.past_key_values)
