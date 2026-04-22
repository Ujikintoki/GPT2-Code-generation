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
from transformers import PreTrainedModel

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
        "--freeze_strategy",
        type=str,
        choices=["none", "freeze_bottom", "freeze_top"],
        default="none",
        help="Layer-wise freezing strategy for ablation study.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    return parser.parse_args()


def apply_layer_freezing(model: PreTrainedModel, strategy: str) -> PreTrainedModel:
    """
    Apply layer-wise freezing to GPT-2 for ablation studies.

    Args:
        model: The initialized GPT-2 model.
        strategy: 'none', 'freeze_bottom', or 'freeze_top'.
    """
    if strategy == "none":
        return model

    logger.info("Applying layer freezing strategy: %s", strategy)

    if strategy == "freeze_bottom":
        # 冻结词嵌入 (Word Embeddings) 和位置嵌入 (Position Embeddings)
        model.transformer.wte.requires_grad_(False)
        model.transformer.wpe.requires_grad_(False)

        # 冻结底层：前 6 层 Transformer Blocks (0 到 5)
        for i in range(6):
            for param in model.transformer.h[i].parameters():
                param.requires_grad = False
        logger.info("Bottom layers (Embeddings + Blocks 0-5) successfully frozen.")

    elif strategy == "freeze_top":
        # 冻结高层：后 6 层 Transformer Blocks (6 到 11)
        for i in range(6, 12):
            for param in model.transformer.h[i].parameters():
                param.requires_grad = False
        logger.info("Top layers (Blocks 6-11) successfully frozen.")

    else:
        raise ValueError(f"Unknown freezing strategy: {strategy}")

    # 计算并记录参数比例，供论文图表使用
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable_params / total_params

    logger.info(
        "Trainable parameters: %d / %d (%.2f%%)",
        trainable_params,
        total_params,
        percentage,
    )

    return model


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

    # 根据 CLI 参数应用层级冻结策略
    model = apply_layer_freezing(model, strategy=args.freeze_strategy)

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
        eval_strategy="epoch",
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
