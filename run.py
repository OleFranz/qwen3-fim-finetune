from unsloth import FastLanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizer

model: PreTrainedModel
tokenizer: PreTrainedTokenizer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen-fim-final",
    load_in_4bit=True,
    device_map="auto"
)

prefix = "def add(a, b):\n"
suffix = "    return result"
prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
middle = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

GREEN = "\x1b[32m"
RESET = "\x1b[0m"

print(f"Completed code:\n{prefix}{GREEN}{middle}{RESET}{suffix}\n--------------")

raw = tokenizer.decode(outputs[0])

print(f"\nRaw:\n{raw!r}\n---")