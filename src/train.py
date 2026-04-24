"""
Domain-Adaptive Fine-Tuning of Causal Language Models.

This script executes the fine-tuning of GPT-family models on a Python code dataset.
It serves as a unified entry point, dynamically adjusting training hyperparameters
and optimizers based on the target hardware architecture (e.g., T4 vs. A100)
to ensure optimal VRAM utilization and computational efficiency.
"""

import argparse
import logging
import math
import os
import sys

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
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for unified training configuration."""
    parser = argparse.ArgumentParser(description="Unified Causal LM Fine-Tuning")

    # Path configurations
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the processed Hugging Face dataset.",
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
        default="./models/unified-model-final",
        help="Directory to save the final model artifacts.",
    )

    # Model and Hardware Configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        choices=["gpt2", "distilgpt2", "gpt2-medium"],
        help="Base model architecture to initialize.",
    )
    parser.add_argument(
        "--gpu_target",
        type=str,
        default="T4",
        choices=["T4", "A100"],
        help="Target hardware architecture for dynamic hyperparameter routing.",
    )

    # General Hyperparameters (Defaults act as a baseline, dynamically overridden)
    parser.add_argument(
        "--epochs", type=int, default=3, help="Total number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Baseline batch size per device (may be overridden by gpu_target).",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Baseline gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Initial learning rate for the optimizer.",
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


def configure_hardware_profile(args: argparse.Namespace) -> dict:
    """
    Dynamically route hardware-specific training arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        dict: A dictionary of kwargs to pass to TrainingArguments.
    """
    profile = {}

    if args.gpu_target == "A100":
        logger.info("Applying A100 hardware profile: high throughput, fused optimizer.")
        profile["per_device_train_batch_size"] = args.batch_size
        profile["per_device_eval_batch_size"] = args.batch_size
        profile["gradient_accumulation_steps"] = min(
            2, args.grad_accum
        )  # Reduce grad accum for A100
        profile["optim"] = "adamw_torch_fused"
        # A100 supports bfloat16, which is vastly superior for LLM training stability
        profile["bf16"] = torch.cuda.is_bf16_supported()
        profile["fp16"] = not profile["bf16"]
    else:
        logger.info("Applying T4 hardware profile: memory-constrained optimization.")
        profile["per_device_train_batch_size"] = min(
            8, args.batch_size
        )  # Strict constraint
        profile["per_device_eval_batch_size"] = min(8, args.batch_size)
        profile["gradient_accumulation_steps"] = max(
            4, args.grad_accum
        )  # Ensure large effective batch size
        profile["optim"] = "adamw_torch"
        profile["fp16"] = torch.cuda.is_available()
        profile["bf16"] = False

    return profile


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    logger.info("Initializing unified training pipeline.")
    logger.info("Model: %s | Hardware Target: %s", args.model_name, args.gpu_target)

    # 1. Load and prepare dataset
    logger.info("Loading dataset from %s", args.data_path)
    try:
        dataset = load_from_disk(args.data_path)
    except Exception as e:
        logger.error("Failed to load dataset: %s", str(e))
        sys.exit(1)

    if args.data_fraction < 1.0:
        subset_size = int(len(dataset) * args.data_fraction)
        dataset = dataset.select(range(subset_size))
        logger.info("Dataset truncated to %d samples.", subset_size)

    split_dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    logger.info(
        "Dataset split complete. Train size: %d, Eval size: %d",
        len(train_dataset),
        len(eval_dataset),
    )

    # 2. Initialize model and tokenizer
    logger.info("Loading architecture: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 3. Dynamic Configuration of TrainingArguments
    hw_profile_kwargs = configure_hardware_profile(args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to="none",
        **hw_profile_kwargs,  # Inject hardware specific settings
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

    # 5. Evaluate and log metrics internally
    logger.info("Running final internal evaluation.")
    eval_results = trainer.evaluate()
    loss = eval_results.get("eval_loss", float("inf"))
    try:
        ppl = math.exp(loss)
    except OverflowError:
        ppl = float("inf")
    logger.info("Final Validation Loss: %.4f | Perplexity (PPL): %.2f", loss, ppl)

    # 6. Save final artifacts
    os.makedirs(args.final_model_dir, exist_ok=True)
    trainer.save_model(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)
    logger.info(
        "Training pipeline complete. Artifacts saved to: %s", args.final_model_dir
    )


if __name__ == "__main__":
    main()
