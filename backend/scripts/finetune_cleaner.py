"""
Fine-tune mT5-small or AraT5 on (noisy_ocr → clean_text) pairs.
Run after generate_training_data.py produces training_data/ocr_clean_pairs.json
Requires: ~2GB RAM, ~30min on CPU for 3000 samples / 3 epochs
"""

from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import json

MODEL_NAME = "google/mt5-small"
OUTPUT_DIR = "weights/ocr_cleaner_finetuned"


def load_data(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    return Dataset.from_list(pairs)


def tokenize(batch, tokenizer, max_len=256):
    model_inputs = tokenizer(
        batch["input"], max_length=max_len,
        truncation=True, padding="max_length"
    )
    labels = tokenizer(
        batch["output"], max_length=64,
        truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    dataset = load_data("training_data/ocr_clean_pairs.json")
    split = dataset.train_test_split(test_size=0.1)

    tokenized = split.map(
        lambda b: tokenize(b, tokenizer),
        batched=True, remove_columns=["input", "output", "field"]
    )

    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=False,
        optim="adafactor",      # much less memory than AdamW
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()

    # Copy shared embeddings to encoder/decoder so save has all keys
    import torch
    if hasattr(model, "shared") and hasattr(model.encoder, "embed_tokens"):
        model.encoder.embed_tokens.weight = torch.nn.Parameter(model.shared.weight.clone())
    if hasattr(model, "shared") and hasattr(model.decoder, "embed_tokens"):
        model.decoder.embed_tokens.weight = torch.nn.Parameter(model.shared.weight.clone())

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuned model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
