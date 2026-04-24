"""
Evaluation script for Domain-Adaptive Fine-Tuned Models.

Computes cross-entropy loss and perplexity on the held-out test set,
and outputs the results in a formal academic table format.
"""

import argparse
import logging
import math
import sys

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Configure standard academic logging format
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation configuration."""
    parser = argparse.ArgumentParser(
        description="Model Evaluation and Academic Report Generation"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/gpt2-python-final",
        help="Path to the fine-tuned model directory.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/processed/gpt2_python_dataset",
        help="Path to the preprocessed dataset.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (must match training seed to isolate test set).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Initiating evaluation pipeline for model: %s", args.model_path)

    # 1. Extract identical Test Set used during training
    logger.info("Loading test dataset from %s", args.data_path)
    dataset = load_from_disk(args.data_path)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    test_dataset = split_dataset["test"]
    logger.info("Test dataset size: %d samples", len(test_dataset))

    # 2. Load fine-tuned model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
    except Exception as e:
        logger.error(
            "Failed to load model from %s. Ensure the path is correct. Error: %s",
            args.model_path,
            str(e),
        )
        return

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 3. Configure evaluation arguments
    eval_args = TrainingArguments(
        output_dir="./models/eval_temp",
        per_device_eval_batch_size=args.batch_size,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    # 4. Execute evaluation
    logger.info("Computing evaluation metrics...")
    eval_results = trainer.evaluate()

    loss = eval_results.get("eval_loss", float("inf"))
    try:
        ppl = math.exp(loss)
    except OverflowError:
        ppl = float("inf")

    # 5. Output results in a formal academic three-line table format
    print("\n")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Value':<25}")
    print("-" * 60)
    print(f"{'Cross-Entropy Loss':<30} | {loss:<25.4f}")
    print(f"{'Perplexity (PPL)':<30} | {ppl:<25.4f}")
    print("=" * 60)
    print("\n")


if __name__ == "__main__":
    main()
