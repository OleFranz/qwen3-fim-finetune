from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen-chat-fim-final",
    max_seq_length=1024,
    load_in_4bit=False,
    device_map="auto"
)

model.save_pretrained_merged(
    "qwen-chat-fim-merged",
    tokenizer,
    save_method = "merged_16bit"
)

# Then run: (in project root directory)
# git clone https://github.com/ggerganov/llama.cpp
# cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
# cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli
# cp llama.cpp/build/bin/llama-* llama.cpp
# mkdir qwen-chat-fim-gguf
# python llama.cpp/convert_hf_to_gguf.py "qwen-chat-fim-merged" --outfile "qwen-chat-fim-gguf/qwen-chat-fim-gguf-f16.gguf" --outtype f16
# .\llama.cpp\build\bin\Release\llama-quantize.exe qwen-chat-fim-gguf/qwen-chat-fim-gguf-f16.gguf qwen-chat-fim-gguf/qwen-chat-fim-gguf-q4_k_m.gguf q4_k_m
# cd qwen-chat-fim-gguf
# rm qwen-chat-fim-gguf-f16.gguf
# > create "Modelfile" with content:
# > FROM ./qwen-chat-fim-gguf-q4_k_m.gguf
# > TEMPLATE {{- if .Suffix }}<|fim_prefix|>{{ .Prompt }}<|fim_suffix|>{{ .Suffix }}<|fim_middle|>{{ else }}{{ .Prompt }}{{ end }}
# ollama create qwen-chat-fim-gguf -f qwen-chat-fim-gguf/Modelfile