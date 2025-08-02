# qwen3-fim-finetune
Scripts to fine-tune Qwen3 0.6B Base on fill-in-the-middle (FIM) tasks using Unsloth

Made for Windows and Python 3.12.
\
Requires about 20GB of disk space and 4GB of VRAM.

Model used for fine-tuning:
[unsloth/Qwen3-0.6B-Base](https://huggingface.co/unsloth/Qwen3-0.6B-Base)
\
Dataset used for training:
[Orion-zhen/fim-code](https://huggingface.co/datasets/Orion-zhen/fim-code)

The model is trained on the Qwen2.5-Coder FIM template:
\
`<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|endoftext|>`