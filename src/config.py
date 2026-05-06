"""Unified, heavily-typed configuration dataclasses for domain-adaptive fine-tuning.

This module defines the canonical configuration schema used throughout the entire
codebase. All hyperparameters, resource profiles, and data pipeline settings are
declared here as frozen dataclasses with strict type hints (PEP 484).

Usage:
    Train config via CLI:
        python -m src.cli train --config_path configs/training/low_vram.yaml

    Programmatic construction:
        from src.config import TrainingConfig, HardwareConfig, ResourceTier
        hw = HardwareConfig.from_resource_tier(ResourceTier.LOW_VRAM)
        cfg = TrainingConfig(
            model_name="gpt2-medium",
            data_path="./data/processed/gpt2_python_dataset",
            hardware=hw,
        )
        cfg.to_training_arguments()  # -> transformers.TrainingArguments

    Integration with HfArgumentParser:
        from transformers import HfArgumentParser
        parser = HfArgumentParser((TrainingConfig,))
        cfg, = parser.parse_args_into_dataclasses()
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import TrainingArguments as HfTrainingArguments


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ResourceTier(str, Enum):
    """Hardware resource tiers that map to pre-built accelerator profiles.

    Members:
        DEFAULT: Balanced defaults suitable for most mid-range GPUs.
        LOW_VRAM: Optimized for constrained GPU memory (≤16 GB).
            Enables gradient checkpointing, minimises batch sizes, and uses fp16.
        HIGH_THROUGHPUT: Optimized for large-memory, high-compute GPUs (≥40 GB).
            Maximises per-device batch size, minimises gradient accumulation,
            favours bf16 precision, and scales dataloader workers.
    """

    DEFAULT = "default"
    LOW_VRAM = "low_vram"
    HIGH_THROUGHPUT = "high_throughput"


class SchedulerType(str, Enum):
    """Learning-rate scheduler names compatible with HF TrainingArguments."""

    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class OptimizerType(str, Enum):
    """Optimizer backends supported by Hugging Face Trainers."""

    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"


# ---------------------------------------------------------------------------
# Hardware configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareConfig:
    """Resource-oriented hardware profile completely abstracting device details.

    Never reference a specific GPU name (e.g. "T4", "A100") in training code.
    Instead, select a ResourceTier and let this dataclass supply the correct
    hyperparameters.

    Attributes:
        resource_tier: The symbolic resource tier.
        per_device_train_batch_size: Samples per device per forward pass.
        per_device_eval_batch_size: Samples per device during evaluation.
        gradient_accumulation_steps: Number of forward passes before a backward
            pass. The *effective* batch size is
            per_device_train_batch_size × gradient_accumulation_steps × world_size.
        use_fp16: Enable automatic mixed precision (16-bit floats).
        use_bf16: Enable bfloat16 mixed precision (preferred on Ampere+ GPUs).
        use_gradient_checkpointing: Trade compute for memory by recomputing
            activations during the backward pass.
        optimizer: The optimizer backend.
        dataloader_num_workers: Number of subprocesses for data loading.
        dataloader_pin_memory: Pin data loader tensors to CPU memory for faster
            GPU transfer.
        tf32: Enable NVIDIA TensorFloat-32 mode on Ampere+ GPUs.
    """

    resource_tier: ResourceTier = ResourceTier.DEFAULT
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    use_fp16: bool = False
    use_bf16: bool = False
    use_gradient_checkpointing: bool = False
    optimizer: OptimizerType = OptimizerType.ADAMW_TORCH
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    tf32: bool = True

    @staticmethod
    def detect_bf16_support() -> bool:
        """Probe whether the current device supports bfloat16 tensor cores."""
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    @classmethod
    def from_resource_tier(
        cls,
        tier: ResourceTier,
        *,
        world_size: int = 1,
    ) -> "HardwareConfig":
        """Factory that returns a fully-populated profile for a given tier.

        Args:
            tier: The symbolic resource tier.
            world_size: Total number of devices (GPUs) participating in training.
                Used to adjust gradient accumulation so the effective batch size
                remains constant across tiers.

        Returns:
            A frozen HardwareConfig instance tailored to the requested tier.

        Raises:
            ValueError: If an unknown resource tier is supplied.
        """
        bf16_available = cls.detect_bf16_support()

        if tier == ResourceTier.LOW_VRAM:
            return cls(
                resource_tier=tier,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=max(8 // world_size, 1),
                use_fp16=True,
                use_bf16=False,
                use_gradient_checkpointing=True,
                optimizer=OptimizerType.ADAMW_TORCH,
                dataloader_num_workers=2,
                dataloader_pin_memory=True,
                tf32=False,
            )

        if tier == ResourceTier.HIGH_THROUGHPUT:
            return cls(
                resource_tier=tier,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                gradient_accumulation_steps=max(2 // world_size, 1),
                use_fp16=False,
                use_bf16=bf16_available,
                use_gradient_checkpointing=False,
                optimizer=OptimizerType.ADAMW_TORCH_FUSED,
                dataloader_num_workers=8,
                dataloader_pin_memory=True,
                tf32=bf16_available,
            )

        if tier == ResourceTier.DEFAULT:
            return cls(
                resource_tier=tier,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=max(4 // world_size, 1),
                use_fp16=not bf16_available,
                use_bf16=bf16_available,
                use_gradient_checkpointing=False,
                optimizer=OptimizerType.ADAMW_TORCH,
                dataloader_num_workers=4,
                dataloader_pin_memory=True,
                tf32=True,
            )

        raise ValueError(f"Unknown resource tier: {tier}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary for downstream consumers."""
        return {
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": self.use_fp16,
            "bf16": self.use_bf16,
            "optim": self.optimizer.value,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "tf32": self.tf32,
        }


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """Data pipeline configuration for preprocessing and loading.

    Attributes:
        dataset_name: Hugging Face dataset identifier (e.g. ``code_search_net``).
        dataset_config: Sub-configuration name (e.g. ``python``).
        dataset_split: Which split to load (e.g. ``train[:50%]``).
        max_length: Fixed token length for chunked sequences.
        stride: Sliding-window stride in tokens. Equal to ``max_length`` produces
            non-overlapping sequential chunks.
        preprocessing_num_workers: Parallel workers for dataset.map().
        cache_dir: HF datasets cache directory.
        processed_data_dir: Where to persist the tokenized dataset via
            ``save_to_disk``.
    """

    dataset_name: str = "code_search_net"
    dataset_config: str = "python"
    dataset_split: str = "train[:50%]"
    max_length: int = 256
    stride: int = 256
    preprocessing_num_workers: int = 4
    cache_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "datasets"
        )
    )
    processed_data_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "processed",
        )
    )

    @property
    def processed_dataset_path(self) -> str:
        """Absolute path to the saved arrow dataset directory."""
        return os.path.join(self.processed_data_dir, "gpt2_python_dataset")


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Model architecture and tokenizer configuration.

    Attributes:
        model_name_or_path: HF Hub identifier or local path for the pretrained
            causal LM.
        use_fast_tokenizer: Whether to use the Rust-backed fast tokenizer.
        max_position_embeddings: Override the model's maximum position embeddings
            (useful for context-length ablation studies).
        trust_remote_code: Allow execution of remote code from HF Hub.
    """

    model_name_or_path: str = "gpt2"
    use_fast_tokenizer: bool = True
    max_position_embeddings: Optional[int] = None
    trust_remote_code: bool = False


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    """Master configuration aggregating all sub-configs for a single training run.

    This class is designed to be parsed directly from the CLI via
    ``transformers.HfArgumentParser``. When constructed programmatically, use the
    static factory methods to enforce valid resource profiles.

    Attributes:
        data: Data pipeline settings.
        model: Model architecture settings.
        hardware: Hardware/resource profile.
        output_dir: Directory to write intermediate checkpoints.
        final_model_dir: Directory to save the final merged model artifacts.
        num_train_epochs: Total number of passes over the training set.
        learning_rate: Peak learning rate for the scheduler.
        weight_decay: AdamW weight decay coefficient.
        lr_scheduler_type: Learning-rate schedule shape.
        warmup_ratio: Fraction of total steps to spend on linear warmup.
        max_grad_norm: Gradient clipping threshold (L2 norm).
        seed: Global random seed for reproducibility.
        eval_strategy: Evaluation trigger (``epoch`` or ``steps``).
        save_strategy: Checkpoint save trigger.
        save_total_limit: Maximum number of checkpoints to retain.
        load_best_model_at_end: Restore the best checkpoint after training.
        metric_for_best_model: Metric used to determine "best" checkpoint.
        logging_steps: Log metrics every N steps.
        report_to: Integrations to report metrics to (e.g. ``wandb``, ``none``).
        data_fraction: Fraction [0.0, 1.0] of the training set to use (ablation).
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    output_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output",
            "checkpoints",
        )
    )
    final_model_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output",
            "final_models",
        )
    )
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    lr_scheduler_type: SchedulerType = SchedulerType.COSINE
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    seed: int = 42
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    logging_steps: int = 50
    report_to: str = "none"
    data_fraction: float = 1.0

    def __post_init__(self) -> None:
        """Validate inter-field constraints after dataclass construction."""
        if not (0.0 < self.data_fraction <= 1.0):
            raise ValueError(
                f"data_fraction must be in (0.0, 1.0], got {self.data_fraction}"
            )
        if self.warmup_ratio < 0.0 or self.warmup_ratio >= 1.0:
            raise ValueError(
                f"warmup_ratio must be in [0.0, 1.0), got {self.warmup_ratio}"
            )

    @classmethod
    def from_resource_tier(
        cls,
        tier: ResourceTier,
        *,
        model_name_or_path: str = "gpt2",
        data_path: Optional[str] = None,
        data_fraction: float = 1.0,
        **overrides: Any,
    ) -> "TrainingConfig":
        """Build a complete training config from a resource tier.

        Args:
            tier: The symbolic resource tier.
            model_name_or_path: Pretrained model identifier.
            data_path: Path to the processed arrow dataset. If ``None``, uses
                the default from ``DataConfig``.
            data_fraction: Fraction of data to use (ablation).
            **overrides: Additional keyword arguments passed directly to the
                ``TrainingConfig`` constructor (e.g. ``num_train_epochs=5``).

        Returns:
            A frozen, fully-specified ``TrainingConfig``.
        """
        data_cfg = DataConfig()
        if data_path is not None:
            data_cfg = replace(
                data_cfg,
                processed_data_dir=os.path.dirname(data_path),
            )
        return cls(
            data=data_cfg,
            model=ModelConfig(model_name_or_path=model_name_or_path),
            hardware=HardwareConfig.from_resource_tier(tier),
            data_fraction=data_fraction,
            **overrides,
        )

    def to_training_arguments(self) -> HfTrainingArguments:
        """Convert this config into a Hugging Face ``TrainingArguments`` instance.

        This is the single entry point for creating ``TrainingArguments``.
        It consumes the ``HardwareConfig.to_dict()`` output transparently so
        that core training code never touches device-specific logic.

        Returns:
            A fully populated ``transformers.TrainingArguments`` object ready
            to be passed to ``Trainer``.
        """
        return HfTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type.value,
            warmup_ratio=self.warmup_ratio,
            max_grad_norm=self.max_grad_norm,
            seed=self.seed,
            eval_strategy=self.eval_strategy,
            save_strategy=self.save_strategy,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            logging_steps=self.logging_steps,
            report_to=self.report_to.split(",") if self.report_to != "none" else [],
            **self.hardware.to_dict(),
        )

    def effective_batch_size(self, world_size: int = 1) -> int:
        """Compute the global effective batch size.

        Args:
            world_size: Number of devices participating in the training run.

        Returns:
            effective_batch_size = per_device_batch × grad_accum × world_size.
        """
        return (
            self.hardware.per_device_train_batch_size
            * self.hardware.gradient_accumulation_steps
            * world_size
        )


# ---------------------------------------------------------------------------
# Ablation study configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AblationConfig:
    """Configuration for data-scaling and model-capacity ablation studies.

    Attributes:
        base_config: The fully-populated training configuration to ablate from.
        data_fractions: List of dataset fractions to sweep over.
        model_names: List of model architectures to compare.
        seeds: Random seeds for each trial.
    """

    base_config: TrainingConfig
    data_fractions: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
    model_names: Tuple[str, ...] = ("gpt2", "distilgpt2", "gpt2-medium")
    seeds: Tuple[int, ...] = (42, 123, 456)

    def iter_runs(self) -> List[Dict[str, Any]]:
        """Generate a flat list of all ablation trial configurations.

        Returns:
            A list of dictionaries, each containing ``model_name``,
            ``data_fraction``, ``seed``, and a frozen ``TrainingConfig``
            for that specific trial.
        """
        runs: List[Dict[str, Any]] = []
        for model_name in self.model_names:
            for fraction in self.data_fractions:
                for seed in self.seeds:
                    trial_cfg = replace(
                        self.base_config,
                        model=replace(self.base_config.model, model_name_or_path=model_name),
                        data_fraction=fraction,
                        seed=seed,
                    )
                    runs.append(
                        {
                            "model_name": model_name,
                            "data_fraction": fraction,
                            "seed": seed,
                            "config": trial_cfg,
                        }
                    )
        return runs


# ---------------------------------------------------------------------------
# Top-level aggregate (for potential YAML serialization)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectConfig:
    """Aggregate project-level configuration.

    This can be serialized to/from YAML for experiment tracking while keeping
    all sub-configs fully typed.

    Attributes:
        training: The active training configuration.
        experiment_name: A human-readable name for logging and artifact folders.
        tags: Optional list of string tags for experiment tracking.
        notes: Free-form notes about this experiment run.
    """

    training: TrainingConfig
    experiment_name: str = "gpt2-code-finetuning"
    tags: List[str] = field(default_factory=list)
    notes: str = ""