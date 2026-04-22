"""
Domain-Adaptive Fine-Tuning of Causal Language Models.

This script executes the fine-tuning of GPT-2 on a Python code dataset,
optimized for environments with constrained VRAM (e.g., NVIDIA T4 16GB).
"""

import argparse
import logging
import os
import torch

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Configure standard academic logging format
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="GPT-2 Domain-Adaptive Fine-Tuning")

    # Path configurations
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/processed/gpt2_python_dataset",
        help="Path to the processed dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/checkpoints",
        help="Directory to save intermediate checkpoints.",
    )
    parser.add_argument(
        "--final_model_dir",
        type=str,
        default="./models/gpt2-python-final",
        help="Directory to save the final model.",
    )

    # Hyperparameters (Defaulted to T4 optimal settings)
    parser.add_argument(
        "--epochs", type=int, default=3, help="Total number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per device during training.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Initial learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use for ablation studies (0.0 to 1.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    logger.info(
        "Initializing training pipeline. Data fraction set to: %.2f", args.data_fraction
    )

    # 1. Load and prepare dataset
    logger.info("Loading dataset from %s", args.data_path)
    dataset = load_from_disk(args.data_path)

    if args.data_fraction < 1.0:
        subset_size = int(len(dataset) * args.data_fraction)
        dataset = dataset.select(range(subset_size))
        logger.info("Dataset truncated to %d samples for ablation study.", subset_size)

    # Ensure consistent train/test split via fixed seed
    split_dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    logger.info(
        "Dataset split complete. Train size: %d, Eval size: %d",
        len(train_dataset),
        len(eval_dataset),
    )

    # 2. Initialize model and tokenizer
    logger.info("Loading pre-trained GPT-2 model and tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 3. Configure TrainingArguments
    use_fp16 = torch.cuda.is_available()
    if use_fp16:
        logger.info("CUDA detected. FP16 mixed precision training enabled.")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=use_fp16,
        eval_strategy="epoch",  # <--- 已修复为最新版 API 规范
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 4. Execute training process
    logger.info("Commencing fine-tuning process.")
    trainer.train()

    # 5. Save final artifacts
    os.makedirs(args.final_model_dir, exist_ok=True)
    trainer.save_model(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)
    logger.info(
        "Training complete. Best model artifacts saved to: %s", args.final_model_dir
    )


if __name__ == "__main__":
    main()
