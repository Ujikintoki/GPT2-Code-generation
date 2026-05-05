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
    # Create argument parser for GPT2 model evaluation
    parser = argparse.ArgumentParser(description="Evaluate the performance of pre-trained GPT2 language models")
    parser.add_argument("--model_path", type=str, required=True, help="Local file path of the trained GPT2 model")
    parser.add_argument("--data_path", type=str, required=True, help="Local file path of the processed test dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for model evaluation inference")
    args = parser.parse_args()

    # Load processed test dataset from local disk
    print(f"Loading test dataset from local path: {args.data_path}")
    test_dataset = load_from_disk(args.data_path)

    # Load pre-trained model and tokenizer with local file only mode
    print(f"Loading trained GPT2 model and tokenizer from local path: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    # Set pad token equal to eos token for GPT2 (no default pad token)
    tokenizer.pad_token = tokenizer.eos_token

    # Load causal language model for GPT2
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)

    # Initialize data collator for causal language modeling (no masked language modeling)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set clean and stable evaluation training arguments
    evaluation_args = TrainingArguments(
        output_dir="./evaluation_temp_output",
        per_device_eval_batch_size=args.batch_size,
        fp16=True if torch.cuda.is_available() else False,
        report_to="none",
        do_train=False,
        do_eval=True
    )

    # Initialize Hugging Face Trainer for evaluation only
    trainer = Trainer(
        model=model,
        args=evaluation_args,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    # Run model evaluation on test dataset
    evaluation_results = trainer.evaluate()
    test_loss = evaluation_results["eval_loss"]
    # Calculate perplexity from evaluation loss
    perplexity_score = math.exp(test_loss)

    # Print academic standard evaluation result table
    print("\n" + "="*60)
    print(f"{'Evaluation Metric':<30} | {'Final Calculated Value':<25}")
    print("-"*60)
    print(f"{'Test Set Loss':<30} | {test_loss:<25.4f}")
    print(f"{'Perplexity (PPL)':<30} | {perplexity_score:<25.2f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
