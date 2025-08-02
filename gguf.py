from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen-fim-final",
    max_seq_length=1024,
    load_in_4bit=False,
    device_map="auto"
)

model.save_pretrained_merged(
    "qwen-fim-merged",
    tokenizer,
    save_method = "merged_16bit"
)

# Then run:
# git clone https://github.com/ggerganov/llama.cpp
# cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
# cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli
# cp llama.cpp/build/bin/llama-* llama.cpp
# mkdir qwen-fim-gguf
# python llama.cpp/convert_hf_to_gguf.py "qwen-fim-merged" --outfile "qwen-fim-gguf/qwen-fim-gguf-f16.gguf" --outtype f16
# .\llama.cpp\build\bin\Release\llama-quantize.exe qwen-fim-gguf/qwen-fim-gguf-f16.gguf qwen-fim-gguf/qwen-fim-gguf-q4_k_m.gguf q4_k_m
# cd qwen-fim-gguf
# rm qwen-fim-gguf-f16.gguf
# ollama create qwen-fim-gguf -f qwen-fim-gguf/Modelfile