from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer - using a smaller, open-source model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B parameter model, no approval needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Format chat prompt
prompt = """<|system|>
You are a helpful AI assistant.
</s>
<|user|>
What is the capital of France?
</s>
<|assistant|>"""

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Decode and print response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nFull conversation:")
print(response)
print("\nModel's response:")
print(response.split("<|assistant|>")[-1].strip())
