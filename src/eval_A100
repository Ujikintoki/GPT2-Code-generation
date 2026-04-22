import argparse
import math
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GPT2 family models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to processed dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = load_from_disk(args.data_path)
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    test_dataset = dataset["test"]

    # Load model & tokenizer
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Eval arguments
    eval_args = TrainingArguments(
        output_dir="./eval_temp",
        per_device_eval_batch_size=args.batch_size,
        fp16=True if torch.cuda.is_available() else False,
        report_to="none",
        do_train=False,
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    # Evaluate
    results = trainer.evaluate()
    eval_loss = results["eval_loss"]
    perplexity = math.exp(eval_loss)

    # Print academic-style table
    print("\n" + "="*60)
    print(f"{'Metric':<30} | {'Value':<25}")
    print("-"*60)
    print(f"{'Test Loss':<30} | {eval_loss:<25.4f}")
    print(f"{'Perplexity (PPL)':<30} | {perplexity:<25.2f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
