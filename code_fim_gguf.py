from unsloth import FastLanguageModel
import shutil
import os


script_path = str(os.path.dirname(os.path.realpath(__file__))).replace("\\", "/")
if script_path[-1] != "/": script_path += "/"

if os.path.exists(f"{script_path}qwen-code-fim-merged"):
    shutil.rmtree(f"{script_path}qwen-code-fim-merged")


from transformers import PreTrainedModel, PreTrainedTokenizer

model: PreTrainedModel
tokenizer: PreTrainedTokenizer


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen-code-fim-final",
    max_seq_length=1024,
    load_in_4bit=False,
    device_map="auto"
)

model.save_pretrained_merged(
    "qwen-code-fim-merged",
    tokenizer,
    save_method="merged_16bit"
)


if os.path.exists(f"{script_path}llama.cpp") == False:
    os.system(f"cd {script_path} && git clone https://github.com/ggerganov/llama.cpp")
    os.system(f"cd {script_path} && cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF")
    os.system(f"cd {script_path} && cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli")
    os.system(f"cd {script_path} && cp llama.cpp/build/bin/llama-* llama.cpp")

if os.path.exists(f"{script_path}qwen-code-fim-gguf"):
    shutil.rmtree(f"{script_path}qwen-code-fim-gguf")

os.mkdir(f"{script_path}qwen-code-fim-gguf")

os.system(f"cd {script_path} && python llama.cpp/convert_hf_to_gguf.py qwen-code-fim-merged --outfile qwen-code-fim-gguf/qwen-code-fim-gguf-f16.gguf --outtype f16")
os.system(f"cd {script_path} && .\\llama.cpp\\build\\bin\\Release\\llama-quantize.exe qwen-code-fim-gguf/qwen-code-fim-gguf-f16.gguf qwen-code-fim-gguf/qwen-code-fim-gguf-q4_k_m.gguf q4_k_m")

os.remove(f"{script_path}qwen-code-fim-gguf/qwen-code-fim-gguf-f16.gguf")

with open(f"{script_path}qwen-code-fim-gguf/Modelfile", "w") as f:
    f.write("""
FROM ./qwen-code-fim-gguf-q4_k_m.gguf
TEMPLATE {{- if .Suffix }}<|fim_prefix|>{{ .Prompt }}<|fim_suffix|>{{ .Suffix }}<|fim_middle|>{{ else }}{{ .Prompt }}{{ end }}
""".strip())

os.system("ollama rm qwen-code-fim-gguf")
os.system("ollama create qwen-code-fim-gguf -f qwen-code-fim-gguf/Modelfile")