# qwen3-fim-finetune
Scripts to fine-tune Qwen3 0.6B Base on fill-in-the-middle (FIM) tasks using Unsloth

Made for Windows and Python 3.12.
\
Requires about 20GB of disk space and 4GB of VRAM.


**Chat FIM:** (*Doesnt work, either bugged train code or too little train data*)
\
Model used for fine-tuning:
[unsloth/Qwen3-0.6B-Base](https://huggingface.co/unsloth/Qwen3-0.6B-Base)
\
Dataset used for training:
[agentlans/high-quality-english-sentences](https://huggingface.co/datasets/agentlans/high-quality-english-sentences)


**Code FIM:**
\
Model used for fine-tuning:
[unsloth/Qwen3-0.6B-Base](https://huggingface.co/unsloth/Qwen3-0.6B-Base)
\
Dataset used for training:
[Orion-zhen/fim-code](https://huggingface.co/datasets/Orion-zhen/fim-code)


The models are trained on the Qwen2.5-Coder FIM template:
\
`<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|endoftext|>`


**INFO**
\
You might need to run the scripts with UTF-8 mode enabled to avoid encoding issues on Windows:
\
`python -X utf8 {...}.py`
\
Or set the system variable `PYTHONUTF8=1` and restart your terminal.