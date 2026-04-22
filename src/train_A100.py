import os
import math
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT2 on Python Code")
    parser.add_argument("--data_path", type=str, default=None, help="Path to processed dataset")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/GPT2-Code-generation/models/gpt2-code", help="Save model path")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of data to use")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name: gpt2, distilgpt2, gpt2-medium")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading dataset from {args.data_path}")
    lm_datasets = load_from_disk(args.data_path)

    if args.data_fraction < 1.0:
        lm_datasets = lm_datasets.select(range(int(len(lm_datasets) * args.data_fraction)))

    lm_datasets = lm_datasets.train_test_split(test_size=0.1, seed=42)
    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["test"]

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ========== 已修复所有参数错误 ==========
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=True,
        optim="adamw_torch_fused",
        eval_strategy="epoch",  # 已修正
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Validation Loss: {eval_results['eval_loss']}")
    try:
        perplexity = math.exp(eval_results["eval_loss"])
        print(f"Validation Perplexity (PPL): {perplexity:.2f}")
    except OverflowError:
        print("Perplexity inf (loss too high)")

    final_model_path = os.path.join(args.output_dir, "final-model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Best model saved to {final_model_path}")

if __name__ == "__main__":
    main()
