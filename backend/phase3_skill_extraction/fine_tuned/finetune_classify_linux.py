"""
Fine-tune Llama-3.1-8B for page classification using Unsloth.
Optimized for RTX 5090 (32GB VRAM).
Trains on classification_dataset.jsonl to classify pages as single_job_detail or not.

Usage:
    python finetune_classify_linux.py [--epochs 3] [--lr 2e-5]

Prerequisites:
    pip install unsloth
    python build_classification_dataset.py  (to generate dataset first)
"""

import json
import os
import random

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments


# ============== Configuration ==============
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
DATASET_FILE = "fine_tuned_dataset_classify_card_detail.jsonl"
OUTPUT_DIR = "classify_output_linux"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "adapters")

# Training hyperparameters - Optimized for RTX 5090 (32GB VRAM)
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0
MAX_SEQ_LENGTH = 4096
WARMUP_STEPS = 50
SAVE_STEPS = 50
EVAL_STEPS = 50
LOGGING_STEPS = 10

# Llama 3.1 chat template
LLAMA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% set system_message = false %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if loop.index0 == 0 and system_message != false %}"
    "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + system_message + '<|eot_id|>' }}"
    "{% endif %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)


# ============== Dataset Preparation ==============
def convert_base_format_to_messages(record: dict) -> dict:
    """Convert base format to messages format for training."""
    page_text = record["input"]
    is_single = record["output"]["is_single_job_detail"]

    user_prompt = f"""Does this page contain ONE main job posting with a detailed description?

Rules:
- TRUE: Page has ONE main job with detailed description/requirements. Ignore any "Recommended Jobs" or "Similar Jobs" sections.
- FALSE: Page is a job listing/search results showing MULTIPLE jobs as the main content.

Return ONLY JSON: {{"is_single_job_detail": true}} or {{"is_single_job_detail": false}}

Page content:
{page_text}"""

    assistant_response = json.dumps({"is_single_job_detail": is_single})

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def load_dataset_from_jsonl():
    """Load and split the classification dataset."""
    print(f"Reading dataset from {DATASET_FILE}...")

    examples = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if "input" in record and "output" in record:
                    converted = convert_base_format_to_messages(record)
                    examples.append(converted)
                elif "messages" in record:
                    examples.append({"messages": record["messages"]})
            except json.JSONDecodeError:
                continue

    print(f"Total examples: {len(examples)}")

    true_count = sum(
        1 for ex in examples
        if '"is_single_job_detail": true' in ex["messages"][-1]["content"]
    )
    false_count = len(examples) - true_count
    print(f"Labels: TRUE={true_count}, FALSE={false_count}")

    random.seed(42)
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    train_data = examples[:train_end]
    valid_data = examples[train_end:valid_end]
    test_data = examples[valid_end:]

    print(f"Split: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")

    return {
        "train": Dataset.from_list(train_data),
        "valid": Dataset.from_list(valid_data),
        "test": Dataset.from_list(test_data),
    }


# ============== Training ==============
def train(epochs=None, lr=None):
    """Fine-tune the model using Unsloth."""
    epochs = epochs or EPOCHS
    lr = lr or LEARNING_RATE

    print(f"\n{'='*60}")
    print(f"Starting Llama-3.1-8B Fine-tuning (Unsloth)")
    print(f"Optimized for RTX 5090")
    print(f"{'='*60}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {BATCH_SIZE} (grad accum: {GRADIENT_ACCUMULATION_STEPS})")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"Output: {ADAPTER_DIR}")
    print(f"{'='*60}\n")

    # Load dataset
    datasets = load_dataset_from_jsonl()

    # Load model and tokenizer with Unsloth
    print("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,  # 4-bit quantization to fit in 32GB VRAM
    )

    # Set chat template for Llama 3.1
    if tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA_CHAT_TEMPLATE
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Formatting function using the tokenizer
    def formatting_func(examples):
        msgs = examples["messages"]
        # Handle single example (list of dicts) vs batched (list of lists)
        if isinstance(msgs[0], dict):
            # Single example: msgs is [{"role":..}, {"role":..}]
            return [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)]
        # Batched: msgs is [[{"role":..}, ...], [{"role":..}, ...]]
        output = []
        for messages in msgs:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            output.append(text)
        return output

    # Apply LoRA with Unsloth
    print("Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=lr,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        bf16=True,
        fp16=False,
        report_to="none",
        seed=42,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        formatting_func=formatting_func,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()

    # Save the final model
    print(f"\nSaving adapter to {ADAPTER_DIR}...")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Adapters saved to: {ADAPTER_DIR}")
    print(f"{'='*60}")

    # Test inference
    print("\nTesting inference on validation set...")
    test_inference(model, tokenizer, datasets["valid"])

    return 0


def test_inference(model, tokenizer, test_dataset, num_samples=3):
    """Test the fine-tuned model on a few samples."""
    import torch
    FastLanguageModel.for_inference(model)

    print(f"\nTesting on {num_samples} random samples:")
    print("=" * 60)

    samples = random.sample(list(test_dataset), min(num_samples, len(test_dataset)))

    for i, sample in enumerate(samples, 1):
        messages = sample["messages"]
        user_msg = messages[0]["content"]
        expected = messages[1]["content"]

        prompt = tokenizer.apply_chat_template(
            [messages[0]],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
            )

        prediction = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print(f"\nSample {i}:")
        print(f"Input: {user_msg[:100]}...")
        print(f"Expected: {expected}")
        print(f"Predicted: {prediction}")
        print("-" * 60)


# ============== Main ==============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.1-8B for classification")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE})")

    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr)
