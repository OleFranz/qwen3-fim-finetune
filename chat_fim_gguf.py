from unsloth import FastLanguageModel
import shutil
import os


script_path = str(os.path.dirname(os.path.realpath(__file__))).replace("\\", "/")
if script_path[-1] != "/": script_path += "/"

if os.path.exists(f"{script_path}qwen-chat-fim-merged"):
    shutil.rmtree(f"{script_path}qwen-chat-fim-merged")


from transformers import PreTrainedModel, PreTrainedTokenizer

model: PreTrainedModel
tokenizer: PreTrainedTokenizer


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen-chat-fim-final",
    max_seq_length=256,
    load_in_4bit=False,
    device_map="auto"
)

model.save_pretrained_merged(
    "qwen-chat-fim-merged",
    tokenizer,
    save_method="merged_16bit"
)


if os.path.exists(f"{script_path}llama.cpp") == False:
    os.system(f"cd {script_path} && git clone https://github.com/ggerganov/llama.cpp")
    os.system(f"cd {script_path} && cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF")
    os.system(f"cd {script_path} && cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli")
    os.system(f"cd {script_path} && cp llama.cpp/build/bin/llama-* llama.cpp")

if os.path.exists(f"{script_path}qwen-chat-fim-gguf"):
    shutil.rmtree(f"{script_path}qwen-chat-fim-gguf")

os.mkdir(f"{script_path}qwen-chat-fim-gguf")

os.system(f"cd {script_path} && python llama.cpp/convert_hf_to_gguf.py qwen-chat-fim-merged --outfile qwen-chat-fim-gguf/qwen-chat-fim-gguf-f16.gguf --outtype f16")
os.system(f"cd {script_path} && .\\llama.cpp\\build\\bin\\Release\\llama-quantize.exe qwen-chat-fim-gguf/qwen-chat-fim-gguf-f16.gguf qwen-chat-fim-gguf/qwen-chat-fim-gguf-q4_k_m.gguf q4_k_m")

os.remove(f"{script_path}qwen-chat-fim-gguf/qwen-chat-fim-gguf-f16.gguf")

with open(f"{script_path}qwen-chat-fim-gguf/Modelfile", "w") as f:
    f.write("""
FROM ./qwen-chat-fim-gguf-q4_k_m.gguf
TEMPLATE <|fim_prefix|>{{ if .Prompt }}{{ .Prompt }}{{ end }}<|fim_suffix|>{{ if .Suffix }}{{ .Suffix }}{{ end }}<|fim_middle|>
""".strip())

os.system("ollama rm qwen-chat-fim-gguf")
os.system("ollama create qwen-chat-fim-gguf -f qwen-chat-fim-gguf/Modelfile")