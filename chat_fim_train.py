import torch
from unsloth import FastQwen3Model, UnslothTrainer
from transformers import TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset

import random
random.seed(42)

import numpy as np
np.random.seed(42)

import warnings
warnings.filterwarnings("ignore")

model: PreTrainedModel
tokenizer: PreTrainedTokenizer

model, tokenizer = FastQwen3Model.from_pretrained(
    model_name="unsloth/Qwen3-0.6B-Base",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=False
)

model = FastQwen3Model.get_peft_model(
    model,
    random_state=42
)

# custom randint function with exponential distribution towards the lower end
def sample_discrete_exp(a: int, b: int, lambd: float = 0.05) -> int:
    """
    Sample an integer from a discrete exponential distribution over the range [a, b].

    The probability of selecting a value decreases exponentially as the value increases,
    favoring smaller integers in the given range. The rate of decay is controlled by
    the parameter `lambd`.

    Parameters:
    -----------
    a : int
        The lower bound of the range (inclusive).
    b : int
        The upper bound of the range (inclusive).
    lambd : float, optional
        The rate parameter of the exponential distribution (default is 0.05).
        - Higher values of `lambd` result in a steeper decline in probability,
          thus more strongly favoring values near `a`.
        - Lower values of `lambd` make the distribution more uniform,
          reducing the bias towards smaller values.

    Returns:
    --------
    int
        A randomly sampled integer from [a, b] based on the defined exponential probabilities.
    """
    x_values = np.arange(a, b + 1)
    probabilities = np.exp(-lambd * (x_values - a))
    probabilities /= np.sum(probabilities)
    return np.random.choice(x_values, p=probabilities)

def format_fim_prompt(example):
    full_text = example["text"]

    if full_text == None:
        return None

    # tokenize the full text to work with tokens
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    if len(full_tokens) < 10:
        return None

    # choose middle length (1 to 128 tokens)
    middle_length = sample_discrete_exp(1, min(32, len(full_tokens) // 3))

    # calculate available tokens for prefix and suffix
    # 1024 total - 4 special tokens - middle_length = remaining tokens
    remaining_tokens = 1024 - 4 - middle_length

    # divide remaining tokens between prefix and suffix
    prefix_length = remaining_tokens // 2
    suffix_length = remaining_tokens - prefix_length

    # choose a random position for the middle part within the full text
    latest_middle_start = len(full_tokens) - middle_length

    if latest_middle_start < 0:
        return None

    middle_start = random.randint(0, latest_middle_start)
    middle_end = middle_start + middle_length

    # extract new prefix, middle, and suffix tokens from the full text
    all_prefix_tokens = full_tokens[:middle_start]
    new_middle_tokens = full_tokens[middle_start:middle_end]
    all_suffix_tokens = full_tokens[middle_end:]

    # trim prefix and suffix to fit the token budget
    if len(all_prefix_tokens) > prefix_length:
        # take the last prefix_length tokens to maintain context
        new_prefix_tokens = all_prefix_tokens[-prefix_length:]
    else:
        new_prefix_tokens = all_prefix_tokens

    if len(all_suffix_tokens) > suffix_length:
        # take the first suffix_length tokens to maintain context
        new_suffix_tokens = all_suffix_tokens[:suffix_length]
    else:
        new_suffix_tokens = all_suffix_tokens

    # convert the new tokens back to text
    new_prefix_text = tokenizer.decode(new_prefix_tokens, skip_special_tokens=True)
    new_middle_text = tokenizer.decode(new_middle_tokens, skip_special_tokens=True)
    new_suffix_text = tokenizer.decode(new_suffix_tokens, skip_special_tokens=True)

    # create the FIM format prompt using the new extracted parts
    # Qwen2.5 Coder format: <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|endoftext|>
    fim_prompt = f"<|fim_prefix|>{new_prefix_text}<|fim_suffix|>{new_suffix_text}<|fim_middle|>{new_middle_text}<|endoftext|>"

    # tokenize the FIM prompt
    encoded = tokenizer(
        fim_prompt,
        truncation=True,
        max_length=1024,
        padding=False,
        return_tensors=None
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.copy()

    # mask the prefix and suffix parts, only train on predicting the middle
    fim_middle_token_id = tokenizer.convert_tokens_to_ids("<|fim_middle|>")
    endoftext_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    # find the position of <|fim_middle|> token
    try:
        middle_start_idx = input_ids.index(fim_middle_token_id)
        endoftext_idx = input_ids.index(endoftext_token_id)

        # mask everything except the middle part and endoftext token
        labels_masked = [-100] * len(labels)
        labels_masked[middle_start_idx + 1:endoftext_idx + 1] = labels[middle_start_idx + 1:endoftext_idx + 1]
        labels = labels_masked
    except ValueError:
        # if tokens not found, use original labels
        pass

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

dataset = load_dataset("HanxiGuo/BiScope_Data", split="train")
dataset = dataset.shuffle(seed=42)
dataset = dataset.map(
    format_fim_prompt,
    remove_columns=["text"]
)
dataset = dataset.filter(lambda x: x is not None)

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=1024,
    args=TrainingArguments(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=50,
        learning_rate=0.001,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="qwen-code-fim-checkpoint",
        report_to="none"
    )
)

trainer.train()

model.save_pretrained("qwen-chat-fim-final")
tokenizer.save_pretrained("qwen-chat-fim-final")

print("Training completed. Model saved to 'qwen-chat-fim-final'")