from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen-code-fim-final",
    max_seq_length=1024,
    load_in_4bit=False,
    device_map="auto"
)

model.save_pretrained_merged(
    "qwen-code-fim-merged",
    tokenizer,
    save_method = "merged_16bit"
)

# Then run: (in project root directory)
# git clone https://github.com/ggerganov/llama.cpp
# cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
# cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli
# cp llama.cpp/build/bin/llama-* llama.cpp
# mkdir qwen-code-fim-gguf
# python llama.cpp/convert_hf_to_gguf.py "qwen-code-fim-merged" --outfile "qwen-code-fim-gguf/qwen-code-fim-gguf-f16.gguf" --outtype f16
# .\llama.cpp\build\bin\Release\llama-quantize.exe qwen-code-fim-gguf/qwen-code-fim-gguf-f16.gguf qwen-code-fim-gguf/qwen-code-fim-gguf-q4_k_m.gguf q4_k_m
# cd qwen-code-fim-gguf
# rm qwen-code-fim-gguf-f16.gguf
# > create "Modelfile" with content:
# > FROM ./qwen-code-fim-gguf-q4_k_m.gguf
# > TEMPLATE {{- if .Suffix }}<|fim_prefix|>{{ .Prompt }}<|fim_suffix|>{{ .Suffix }}<|fim_middle|>{{ else }}{{ .Prompt }}{{ end }}
# ollama create qwen-code-fim-gguf -f qwen-code-fim-gguf/Modelfile