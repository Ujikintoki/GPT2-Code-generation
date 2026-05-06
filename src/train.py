"""Domain-Adaptive Fine-Tuning of GPT-2 for Python Code Generation — Training Entrypoint.

This module provides the core training orchestration for causal language model
fine-tuning.  It consumes the unified configuration system from ``src.config``,
the structured logger from ``src.utils.logger``, and preprocessed datasets from
``data/processed/`` to produce checked, evaluated model artifacts in ``output/``.

Architecture Overview
---------------------
1. **Config Integration** — Parses CLI arguments via ``argparse`` and constructs
   a frozen ``TrainingConfig``.  Supports ``--resource_tier`` for hardware-agnostic
   execution (low_vram / high_throughput), and ``--data_fraction`` for ablation
   studies.
2. **Model & Tokenizer** — Loads a pretrained ``GPT2LMHeadModel`` and its BPE
   tokenizer.  Applies gradient checkpointing when the resource tier demands it.
3. **Training Engine** — Leverages the Hugging Face ``Trainer`` API with a
   ``DataCollatorForLanguageModeling`` (``mlm=False``).  A custom
   ``compute_metrics`` function computes **Perplexity (PPL)** from validation
   cross-entropy loss.
4. **Robust Checkpointing** — Saves periodic and best-model checkpoints to
   ``output/checkpoints/``, with the final model promoted to
   ``output/final_models/``.  All milestones are written through the project's
   structured logger.

Usage
-----
    # Low-VRAM execution (T4, RTX 3060, etc.)
    python -m src.train --resource_tier low_vram --data_path ./data/processed/gpt2_python_dataset_full

    # High-throughput execution (A100, H100)
    python -m src.train --resource_tier high_throughput --data_path ./data/processed/gpt2_python_dataset_full --model_name gpt2-medium

    # Ablation: 10% data fraction with distilgpt2
    python -m src.train --resource_tier low_vram --data_fraction 0.10 --model_name distilgpt2 --data_path ./data/processed/gpt2_python_dataset_full

    # Programmatic use
    from src.config import TrainingConfig, ResourceTier
    from src.train import run_training
    cfg = TrainingConfig.from_resource_tier(ResourceTier.LOW_VRAM, model_name_or_path="gpt2")
    run_training(cfg)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F  # type: ignore[import-not-found]
from datasets import Dataset, load_from_disk
from transformers import (  # type: ignore[import-not-found]
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    set_seed,
)
from transformers.trainer_callback import (
    TrainerCallback,  # type: ignore[import-not-found]
)

# ---------------------------------------------------------------------------
# Ensure ``src/`` is on the Python path when executed as a script so that
# config and logger imports work regardless of invocation directory.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (  # noqa: E402
    DataConfig,
    HardwareConfig,
    ModelConfig,
    ResourceTier,
    TrainingConfig,
)
from utils.logger import setup_logger  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level logger (configured once in ``main()``)
# ---------------------------------------------------------------------------
logger = setup_logger(__name__, experiment_name="training-entrypoint")


# ===================================================================
# Utility Functions
# ===================================================================


def _build_argument_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the training entrypoint.

    Every argument either maps directly to a ``TrainingConfig`` field or
    provides a shorthand for constructing one (e.g. ``--resource_tier``).
    """
    parser = argparse.ArgumentParser(
        description="Domain-Adaptive Fine-Tuning of GPT-2 for Python Code Generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --resource_tier low_vram --data_path ./data/processed/gpt2_python_dataset_full\n"
            "  %(prog)s --resource_tier high_throughput --model_name gpt2-medium\n"
            "  %(prog)s --resource_tier low_vram --data_fraction 0.10 --epochs 5\n"
        ),
    )

    # ---- Path configuration ----
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the preprocessed Arrow dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory for intermediate checkpoints. "
            "Defaults to <project_root>/output/checkpoints."
        ),
    )
    parser.add_argument(
        "--final_model_dir",
        type=str,
        default=None,
        help=(
            "Directory for the final merged model artifacts. "
            "Defaults to <project_root>/output/final_models."
        ),
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help=(
            "Directory for structured log files. "
            "Defaults to <project_root>/output/logs."
        ),
    )

    # ---- Model configuration ----
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Hugging Face model identifier (default: %(default)s).",
    )

    # ---- Resource tier (hardware-agnostic) ----
    parser.add_argument(
        "--resource_tier",
        type=str,
        default="default",
        choices=[tier.value for tier in ResourceTier],
        help="Hardware resource tier: 'default', 'low_vram', or 'high_throughput'.",
    )

    # ---- Hyperparameters ----
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: %(default)d).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Peak learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay coefficient (default: %(default)s).",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.06,
        help="Fraction of steps for linear warmup (default: %(default)s).",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold (L2 norm, default: %(default)s).",
    )

    # ---- Ablation / data control ----
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use, in (0.0, 1.0] (default: 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)d).",
    )

    # ---- Logging & reporting ----
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log training metrics every N steps (default: %(default)d).",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps"],
        help="Evaluation trigger (default: %(default)s).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to retain (default: %(default)d).",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Comma-separated integrations for metric reporting (e.g. 'wandb,tensorboard').",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="gpt2-code-finetuning",
        help="Human-readable experiment name for log filenames.",
    )

    return parser


# ===================================================================
# Dataset Loading
# ===================================================================


def load_and_split_dataset(
    data_path: str,
    data_fraction: float = 1.0,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Load a preprocessed Arrow dataset and split into train / validation.

    The dataset is expected to be a chunked, tokenized Hugging Face
    ``Dataset`` saved via ``save_to_disk`` by the preprocessing pipeline.

    Args:
        data_path: Absolute or relative path to the Arrow dataset directory.
        data_fraction: Fraction of chunks to retain, in ``(0.0, 1.0]``.
            Fractional subsets are used for scaling-law ablation studies.
        seed: Random seed for subset selection and train/test split.

    Returns:
        A tuple ``(train_dataset, eval_dataset)`` of Hugging Face
        ``Dataset`` instances.

    Raises:
        FileNotFoundError: If ``data_path`` does not exist or is not a
            valid Arrow dataset.
        ValueError: If ``data_fraction`` is out of range.
    """
    if not (0.0 < data_fraction <= 1.0):
        raise ValueError(f"data_fraction must be in (0.0, 1.0], got {data_fraction}")

    logger.info("Loading preprocessed dataset from: %s", data_path)

    dataset = load_from_disk(data_path)
    total_chunks = len(dataset)
    logger.info("Dataset loaded: %d total chunks.", total_chunks)

    # Deterministic fractional subset
    if data_fraction < 1.0:
        subset_size = max(int(total_chunks * data_fraction), 1)
        dataset = dataset.shuffle(seed=seed).select(range(subset_size))
        logger.info(
            "Subsampled %d / %d chunks (%.1f%%) for ablation study.",
            subset_size,
            total_chunks,
            data_fraction * 100,
        )
    else:
        subset_size = total_chunks

    # 90/10 train/validation split
    split_dataset = dataset.train_test_split(test_size=0.1, seed=seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    logger.info(
        "Train / Val split: %d chunks (%.1f%%) / %d chunks (%.1f%%).",
        len(train_dataset),
        100 * len(train_dataset) / max(subset_size, 1),
        len(eval_dataset),
        100 * len(eval_dataset) / max(subset_size, 1),
    )
    return train_dataset, eval_dataset


# ===================================================================
# Model & Tokenizer Initialization
# ===================================================================


def initialise_model_and_tokenizer(
    model_config: ModelConfig,
    hardware: HardwareConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the pretrained causal LM and its tokenizer with hardware-aware tuning.

    Sets ``pad_token = eos_token`` (GPT-2 has no pad token by default) to
    suppress warnings during tokenization.  Enables gradient checkpointing
    when the hardware profile requests it.

    Args:
        model_config: Frozen ``ModelConfig`` specifying architecture and
            tokenizer settings.
        hardware: Frozen ``HardwareConfig`` specifying memory-optimisation
            flags.

    Returns:
        A tuple ``(model, tokenizer)`` where:
            * ``model`` is a ``GPT2LMHeadModel`` (or variant) on the
              appropriate device with gradient checkpointing configured.
            * ``tokenizer`` is a ``PreTrainedTokenizer`` with EOS as pad.
    """
    model_name = model_config.model_name_or_path

    logger.info(
        "Loading tokenizer for: %s (fast=%s)",
        model_name,
        model_config.use_fast_tokenizer,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=model_config.use_fast_tokenizer,
    )

    # GPT-2 does not have a pad token; assign EOS to PAD.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            "pad_token was unset; assigned eos_token ('%s') as pad_token.",
            tokenizer.eos_token,
        )

    logger.info(
        "Tokenizer loaded: vocab_size=%d, pad_token='%s', eos_token='%s'.",
        tokenizer.vocab_size,
        tokenizer.pad_token,
        tokenizer.eos_token,
    )

    logger.info("Loading pretrained model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Override position embeddings if requested (context-length ablation).
    if model_config.max_position_embeddings is not None:
        if hasattr(model.config, "n_positions"):
            original = model.config.n_positions
            model.config.n_positions = model_config.max_position_embeddings
            logger.info(
                "Overrode n_positions: %d → %d.",
                original,
                model_config.max_position_embeddings,
            )
        elif hasattr(model.config, "max_position_embeddings"):
            original = model.config.max_position_embeddings
            model.config.max_position_embeddings = model_config.max_position_embeddings
            logger.info(
                "Overrode max_position_embeddings: %d → %d.",
                original,
                model_config.max_position_embeddings,
            )

    # Gradient checkpointing: trade compute for memory.
    if hardware.use_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on model.")
        elif hasattr(model.config, "gradient_checkpointing"):
            model.config.gradient_checkpointing = True
            logger.info("Gradient checkpointing enabled via model config.")
        else:
            logger.warning(
                "Hardware profile requests gradient checkpointing but model "
                "(%s) does not support it.  Continuing without.",
                type(model).__name__,
            )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded: total_params=%d, trainable_params=%d (%.2f%%).",
        total_params,
        trainable_params,
        100 * trainable_params / total_params if total_params > 0 else 0.0,
    )

    return model, tokenizer


# ===================================================================
# Metric Computation
# ===================================================================


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute validation cross-entropy loss and Perplexity (PPL).

    This function is passed to the Hugging Face ``Trainer`` as the
    ``compute_metrics`` argument.  It receives the raw logits and
    labels for the evaluation set, computes the per-token cross-entropy
    loss, and derives PPL as ``exp(loss)``.

    PPL is the canonical metric for causal language modelling: lower
    values indicate that the model is less "surprised" by held-out
    code, which correlates with better syntax and semantic understanding.

    Args:
        eval_pred: An ``EvalPrediction`` named tuple containing:
            * ``predictions``: Logits tensor of shape
              ``(num_samples, seq_len, vocab_size)`` or
              ``(num_samples * seq_len, vocab_size)``.
            * ``label_ids``: Ground-truth token IDs of matching shape.

    Returns:
        A dictionary with keys:
            * ``eval_loss``: Mean cross-entropy loss (float).
            * ``perplexity``: ``exp(eval_loss)`` — the exponentiated
              average negative log-likelihood.

    Example:
        >>> from transformers import EvalPrediction
        >>> import torch
        >>> logits = torch.randn(2, 256, 50257)  # [batch, seq, vocab]
        >>> labels = torch.randint(0, 50257, (2, 256))
        >>> metrics = compute_metrics(EvalPrediction(predictions=logits, label_ids=labels))
        >>> print(metrics["perplexity"])
        50321.4
    """
    logits: torch.Tensor = eval_pred.predictions  # type: ignore[assignment]
    labels: torch.Tensor = eval_pred.label_ids  # type: ignore[assignment]

    # Convert to tensors if the Trainer passed numpy arrays.
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)

    # Reshape for cross-entropy: [N, vocab] vs [N]
    # logits: [batch_size * seq_len, vocab_size]
    # labels: [batch_size * seq_len]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,  # Standard ignore index for padding
    )

    try:
        ppl = math.exp(loss.item())
    except OverflowError:
        ppl = float("inf")

    return {
        "eval_loss": loss.item(),
        "perplexity": ppl,
    }


# ===================================================================
# Callback: Logging Milestones
# ===================================================================


class EpochLoggingCallback(TrainerCallback):
    """A Hugging Face Trainer callback that logs epoch-level milestones.

    The Trainer already logs step-level metrics via ``logging_steps``,
    but this callback supplements those with compact, structured log
    messages at each evaluation epoch for downstream experiment tracking.
    """

    def __init__(self) -> None:
        super().__init__()
        self._epoch: int = 0

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Log evaluation metrics at the end of each evaluation epoch."""
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss", float("inf"))
        ppl = metrics.get("perplexity", float("inf"))
        epoch = state.epoch if state is not None else self._epoch
        logger.info(
            "Epoch %.2f | eval_loss: %.4f | perplexity: %.2f",
            epoch,
            eval_loss,
            ppl,
        )
        self._epoch += 1


# ===================================================================
# Trainer Setup
# ===================================================================


def setup_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_config: TrainingConfig,
) -> Trainer:
    """Construct a fully configured Hugging Face ``Trainer`` for causal LM.

    This function is the single point where the ``TrainingConfig`` is
    materialised into a ``TrainingArguments`` object and the trainer is
    instantiated with the correct data collator and metrics callback.

    Args:
        model: The pretrained causal LM (e.g. ``GPT2LMHeadModel``).
        tokenizer: The tokenizer matching the model.
        train_dataset: Tokenized training dataset of fixed-length chunks.
        eval_dataset: Tokenized validation dataset.
        training_config: The frozen master configuration.

    Returns:
        A ``Trainer`` instance ready to call ``trainer.train()``.
    """
    # Data collator: no masked language modelling (mlm=False) => causal LM.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Convert config → Hugging Face TrainingArguments.
    training_args = training_config.to_training_arguments()

    # Build the trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EpochLoggingCallback()],
    )

    logger.info("Trainer initialised successfully.")
    logger.info(
        "  Effective batch size: %d  (%d per device × %d grad_accum × %d GPUs)",
        training_config.effective_batch_size(),
        training_config.hardware.per_device_train_batch_size,
        training_config.hardware.gradient_accumulation_steps,
        1,  # world_size=1; adjust if multi-GPU is detected
    )
    precision = (
        "bf16"
        if training_config.hardware.use_bf16
        else ("fp16" if training_config.hardware.use_fp16 else "fp32")
    )
    gc = "on" if training_config.hardware.use_gradient_checkpointing else "off"
    logger.info("  Precision: %s", precision)
    logger.info("  Gradient checkpointing: %s", gc)
    logger.info("  Checkpoint directory: %s", training_config.output_dir)
    logger.info("  Final model directory: %s", training_config.final_model_dir)

    return trainer


# ===================================================================
# High-Level Orchestration
# ===================================================================


def run_training(training_config: TrainingConfig) -> Dict[str, float]:
    """Execute the full training pipeline using a frozen ``TrainingConfig``.

    This function is the programmatic entry point.  It performs dataset
    loading, model initialisation, training, evaluation, and artifact
    persistence — all driven by the supplied configuration.

    Args:
        training_config: A fully-specified frozen ``TrainingConfig``.

    Returns:
        A dictionary containing the final evaluation metrics:
        ``{"eval_loss": ..., "perplexity": ...}``.
    """
    # ---- Seed everything for reproducibility ----
    set_seed(training_config.seed)
    logger.info("Global random seed set to: %d", training_config.seed)

    # ---- 1. Load and split dataset ----
    train_dataset, eval_dataset = load_and_split_dataset(
        data_path=training_config.data.processed_dataset_path,
        data_fraction=training_config.data_fraction,
        seed=training_config.seed,
    )

    # ---- 2. Initialise model and tokenizer ----
    model, tokenizer = initialise_model_and_tokenizer(
        model_config=training_config.model,
        hardware=training_config.hardware,
    )

    # ---- 3. Create trainer ----
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
    )

    # ---- 4. Train ----
    logger.info("=" * 68)
    logger.info(
        "  Commencing fine-tuning — %d epoch(s)", training_config.num_train_epochs
    )
    logger.info("=" * 68)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current state...")
    except Exception as exc:
        logger.exception("Training failed with an unexpected error: %s", exc)
        raise

    # ---- 5. Final evaluation ----
    logger.info("Running final evaluation on validation set...")
    eval_results = trainer.evaluate()
    loss = eval_results.get("eval_loss", float("inf"))
    ppl = eval_results.get("perplexity", float("inf"))

    logger.info(
        "Final — Validation Loss: %.4f | Perplexity (PPL): %.2f",
        loss,
        ppl,
    )

    # ---- 6. Save final model artifacts ----
    final_dir = training_config.final_model_dir
    os.makedirs(final_dir, exist_ok=True)
    logger.info("Saving final model and tokenizer to: %s", final_dir)

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    logger.info("Training pipeline complete. Artifacts saved to: %s", final_dir)

    return {"eval_loss": loss, "perplexity": ppl}


# ===================================================================
# CLI Entry Point
# ===================================================================


def main(argv: Optional[list[str]] = None) -> None:
    """Parse CLI arguments, build a ``TrainingConfig``, and launch training.

    This is the command-line entry point invoked when the script is
    executed directly or via ``python -m src.train``.

    Args:
        argv: Optional argument list (useful for testing).  When ``None``,
            ``sys.argv[1:]`` is used.
    """
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Configure the project logger *once* before any other logging.
    # We reconfigure the root logger so that the file handler uses the
    # correct experiment name and log directory.
    # ------------------------------------------------------------------
    global logger
    logger = setup_logger(
        __name__,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
    )

    # ------------------------------------------------------------------
    # Resolve resource tier
    # ------------------------------------------------------------------
    resource_tier = ResourceTier(args.resource_tier)

    # ------------------------------------------------------------------
    # Build the frozen TrainingConfig
    # ------------------------------------------------------------------
    data_cfg = DataConfig(
        processed_data_dir=os.path.dirname(os.path.abspath(args.data_path)),
    )

    model_cfg = ModelConfig(
        model_name_or_path=args.model_name,
    )

    hardware_cfg = HardwareConfig.from_resource_tier(resource_tier)

    training_config = TrainingConfig(
        data=data_cfg,
        model=model_cfg,
        hardware=hardware_cfg,
        output_dir=(
            args.output_dir
            if args.output_dir is not None
            else TrainingConfig.output_dir.default_factory()  # type: ignore[arg-type]
        ),
        final_model_dir=(
            args.final_model_dir
            if args.final_model_dir is not None
            else TrainingConfig.final_model_dir.default_factory()  # type: ignore[arg-type]
        ),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        eval_strategy=args.eval_strategy,
        save_strategy=args.eval_strategy,  # Match save to eval for best-model selection
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        data_fraction=args.data_fraction,
    )

    # ------------------------------------------------------------------
    # Log the effective configuration
    # ------------------------------------------------------------------
    logger.info("=" * 68)
    logger.info("  Training Configuration Summary")
    logger.info("=" * 68)
    logger.info("  Model:                %s", training_config.model.model_name_or_path)
    logger.info(
        "  Resource Tier:        %s", training_config.hardware.resource_tier.value
    )
    logger.info("  Data Path:            %s", args.data_path)
    logger.info("  Data Fraction:        %.2f", training_config.data_fraction)
    logger.info("  Epochs:               %d", training_config.num_train_epochs)
    logger.info("  Learning Rate:        %.1e", training_config.learning_rate)
    logger.info("  Warmup Ratio:         %.2f", training_config.warmup_ratio)
    logger.info("  Effective Batch Size: %d", training_config.effective_batch_size())
    precision = (
        "bf16"
        if training_config.hardware.use_bf16
        else ("fp16" if training_config.hardware.use_fp16 else "fp32")
    )
    logger.info("  Precision:            %s", precision)
    logger.info("  Seed:                 %d", training_config.seed)
    logger.info("=" * 68)

    # ------------------------------------------------------------------
    # Launch training
    # ------------------------------------------------------------------
    run_training(training_config)


# ===================================================================
# Make the script both importable and directly executable.
# ===================================================================
if __name__ == "__main__":
    main()
