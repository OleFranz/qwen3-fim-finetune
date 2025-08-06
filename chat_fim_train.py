import torch
from unsloth import FastQwen3Model, UnslothTrainer, UnslothTrainingArguments, add_new_tokens
from transformers import PreTrainedModel, PreTrainedTokenizer
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
    max_seq_length=256,
    dtype=None,
    load_in_4bit=False
)

model = FastQwen3Model.get_peft_model(
    model,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head"
    ],
    random_state=42
)

# custom randint function with exponential distribution towards the lower end
def sample_discrete_exp(a: int, b: int, lambd: float = 0.0) -> int:
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
        The rate parameter of the exponential distribution (default is 0.0).
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

    # pre "filter" text
    if full_text is None: return None
    full_text = str(full_text).strip()
    if len(full_text) < 32: return None

    word_boundaries = []
    in_word = False
    for i, ch in enumerate(full_text):
        if not ch.isspace() and not in_word:
            # word start
            start = i
            in_word = True
        elif ch.isspace() and in_word:
            # word end
            word_boundaries.append((start, i))
            in_word = False
    # if text ends with a word, close it
    if in_word:
        word_boundaries.append((start, len(full_text)))

    # no words → abort
    if not word_boundaries:
        return None

    # choose a word: 50/50 random vs last
    if random.randint(0, 1) == 0:
        idx = random.randint(0, len(word_boundaries) - 1)
    else:
        idx = len(word_boundaries) - 1

    middle_start, middle_end = word_boundaries[idx]

    # include leading space/newline with 50/50 chance (otherwise it stays in prefix)
    if middle_start > 0 and full_text[middle_start - 1].isspace():
        if random.randint(0, 1) == 0:
            middle_start -= 1

    prefix_text = full_text[:middle_start]
    suffix_text = full_text[middle_end:]
    middle_text = full_text[middle_start:middle_end]

    # 50/50 chance to further break up the middle word
    if random.randint(0, 1) == 0 and len(middle_text) > 1:
        # decide how many characters to move to prefix (at least 0)
        chars_to_move_prefix = random.randint(0, len(middle_text) - 1)
        # from remaining characters, decide how many to move to suffix (can be 0)
        remaining_chars = len(middle_text) - chars_to_move_prefix
        chars_to_move_suffix = random.randint(0, remaining_chars - 1) if remaining_chars > 1 else 0
        # move characters
        prefix_text += middle_text[:chars_to_move_prefix]
        suffix_text = middle_text[len(middle_text) - chars_to_move_suffix:] + suffix_text
        middle_text = middle_text[chars_to_move_prefix:len(middle_text) - chars_to_move_suffix]

    middle_token_count = len(tokenizer.encode(middle_text))

    # calculate available tokens for prefix and suffix
    # 256 total - 4 special tokens - middle_length = remaining tokens
    remaining_tokens = 256 - 4 - middle_token_count

    # split remaining tokens 50/50
    pref_budget = sample_discrete_exp(0, remaining_tokens // 2, 0.01) # slight bias towards smaller numbers
    suff_budget = sample_discrete_exp(0, remaining_tokens // 2, 0.01) # slight bias towards smaller numbers

    # encode original prefix/suffix
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix_text, add_special_tokens=False)

    # slice to budgets
    new_prefix_tokens = prefix_tokens[-pref_budget:] if pref_budget > 0 else []
    new_suffix_tokens = suffix_tokens[:suff_budget] if suff_budget > 0 else []

    # decode slices
    new_prefix_text = tokenizer.decode(new_prefix_tokens, skip_special_tokens=True)
    new_suffix_text = tokenizer.decode(new_suffix_tokens, skip_special_tokens=True)

    # trim broken leading word in new_prefix_text
    # if it doesn't start on whitespace, cut up to the first space
    if new_prefix_text and not new_prefix_text[0].isspace():
        first_space = new_prefix_text.find(" ")
        if first_space != -1:
            new_prefix_text = new_prefix_text[first_space:]
    # then drop any residual leading whitespace
    new_prefix_text = new_prefix_text.lstrip()

    # trim broken trailing word in new_suffix_text
    # if it doesn't end on whitespace, cut from the last space
    if new_suffix_text and not new_suffix_text[-1].isspace():
        last_space = new_suffix_text.rfind(" ")
        if last_space != -1:
            new_suffix_text = new_suffix_text[:last_space]
    # then drop any residual trailing whitespace
    new_suffix_text = new_suffix_text.rstrip()

    # create the FIM format prompt using the new extracted parts
    # Qwen2.5 Coder format: <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|endoftext|>
    fim_prompt = f"<|fim_prefix|>{new_prefix_text}<|fim_suffix|>{new_suffix_text}<|fim_middle|>{middle_text}<|endoftext|>"

    # tokenize the FIM prompt
    encoded = tokenizer(
        fim_prompt,
        truncation=True,
        max_length=256,
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

dataset = load_dataset("agentlans/high-quality-english-sentences", split="train")
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
    dataset_text_field="text",
    max_seq_length=256,
    args=UnslothTrainingArguments(
        dataset_num_proc=1,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        max_steps=1000,
        learning_rate=0.0002,
        embedding_learning_rate=0.0001,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        torch_empty_cache_steps=10,
        logging_steps=10,
        save_steps=500,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="qwen-chat-fim-checkpoint",
        report_to="none"
    )
)

trainer.train()

model.save_pretrained("qwen-chat-fim-final")
tokenizer.save_pretrained("qwen-chat-fim-final")

print("Training completed. Model saved to 'qwen-chat-fim-final'")