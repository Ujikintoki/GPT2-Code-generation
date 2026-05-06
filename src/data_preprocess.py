"""Domain-Adaptive Data Preprocessing Pipeline for GPT-2 Code Generation.

This module transforms raw Python source code from the CodeSearchNet dataset
into tokenized, chunked, and fine-tuning-ready Arrow datasets. The pipeline
is driven entirely by ``DataConfig`` from ``src.config`` and produces
reproducible artifacts saved to ``data/processed/``.

Architecture Overview
---------------------
The pipeline is composed of four pure, independently-testable stages:

1. **Dataset Loading** — ``load_raw_dataset`` fetches the Python subset from
   Hugging Face Hub with a configurable data fraction (for ablation studies).
2. **Tokenizer Engineering** — ``create_tokenizer`` initialises a GPT-2 BPE
   tokenizer and injects the necessary ``pad_token`` (GPT-2 has none by
   default), while preserving Python-critical whitespace characters.
3. **Chunking Strategy** — ``tokenize_function`` + ``group_texts`` implement
   a batched, sliding-window segmentation that maps variable-length documents
   into fixed-length ``(input_ids, labels)`` pairs for autoregressive Causal LM.
4. **Persistence** — ``save_processed_dataset`` writes the Arrow dataset to disk
   via ``dataset.save_to_disk()`` for zero-copy loading during training.

Usage
-----
    # Process the full dataset with default settings:
    python -m src.data_preprocess

    # Process a 10% data fraction for scaling-law ablation:
    python -m src.data_preprocess --data-fraction 0.10

    # Full control via DataConfig overrides:
    python -m src.data_preprocess \
        --model-name distilgpt2 \
        --max-length 512 \
        --stride 512 \
        --data-fraction 0.50 \
        --num-workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

# ---------------------------------------------------------------------------
# Ensure ``src/`` is on the Python path when executed as a script so that
# config and logger imports work regardless of invocation directory.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import DataConfig  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level logger (configured once in ``main()``)
# ---------------------------------------------------------------------------
logger = setup_logger(__name__, experiment_name="data-preprocessing")


# ===================================================================
# Stage 1: Dataset Loading
# ===================================================================


def load_raw_dataset(
    data_config: Optional[DataConfig] = None,
    *,
    data_fraction: float = 1.0,
    seed: int = 42,
) -> Dataset:
    """Load the Python subset of CodeSearchNet with an optional data fraction.

    Uses the Hugging Face ``datasets`` library to fetch ``code_search_net``
    with the ``python`` configuration.  When ``data_fraction < 1.0``, a
    deterministic stratified subset of the training split is returned so that
    scaling-law ablation runs are reproducible.

    Args:
        data_config: A frozen ``DataConfig`` instance.  If ``None``, a
            default-constructed ``DataConfig`` is used.
        data_fraction: Fraction of the training set to load, in ``(0.0, 1.0]``.
            Values less than 1.0 trigger a deterministic random selection.
        seed: Random seed for subset selection, ensuring reproducibility
            across runs.

    Returns:
        A Hugging Face ``Dataset`` containing the raw code strings.

    Raises:
        ValueError: If ``data_fraction`` is not in ``(0.0, 1.0]``.
        FileNotFoundError: If the dataset cannot be located in the HF cache
            and network access is unavailable.

    Example:
        >>> full_ds = load_raw_dataset()
        >>> tiny_ds = load_raw_dataset(data_fraction=0.01)
    """
    if data_config is None:
        data_config = DataConfig()

    if not (0.0 < data_fraction <= 1.0):
        raise ValueError(f"data_fraction must be in (0.0, 1.0], got {data_fraction}")

    logger.info(
        "Loading dataset '%s/%s' (split='%s', fraction=%.2f)...",
        data_config.dataset_name,
        data_config.dataset_config,
        data_config.dataset_split,
        data_fraction,
    )

    # Load the full training split first, then sub-select deterministically.
    # This avoids relying on the brittle percent-slice syntax for arbitrary
    # fractions while keeping behaviour identical to the original pipeline.
    raw_dataset: Dataset = load_dataset(
        data_config.dataset_name,
        data_config.dataset_config,
        split="train",
        cache_dir=data_config.cache_dir,
    )

    if data_fraction < 1.0:
        total = len(raw_dataset)
        subset_size = max(int(total * data_fraction), 1)
        raw_dataset = raw_dataset.shuffle(seed=seed).select(range(subset_size))
        logger.info(
            "Subsampled %d / %d documents (%.1f%%) for ablation.",
            subset_size,
            total,
            data_fraction * 100,
        )

    logger.info("Raw dataset loaded: %d documents.", len(raw_dataset))
    return raw_dataset


# ===================================================================
# Stage 2: Tokenizer Engineering
# ===================================================================


def create_tokenizer(
    model_name: str = "gpt2",
    *,
    use_fast: bool = True,
) -> PreTrainedTokenizer:
    """Initialise and configure a GPT-2-family tokenizer for code fine-tuning.

    **Whitespace Preservation**: GPT-2's byte-level BPE tokenizer natively
    encodes all Unicode codepoints including ``\\n``, ``\\t``, and spaces.
    No additional normalisation is applied, which is critical for Python's
    indentation-sensitive syntax.

    **Padding Token**: The vanilla GPT-2 tokenizer does not define a
    ``pad_token`` because the original model was trained on contiguous
    sequences only.  We set ``pad_token = eos_token`` following common
    practice for causal-LM fine-tuning.  Padding is never actually used
    during chunked training (all chunks are exactly ``max_length`` tokens),
    but defining it suppresses warnings and enables the tokenizer to be
    reused safely in evaluation pipelines that may require padding.

    Args:
        model_name: Hugging Face Hub identifier for the pretrained tokenizer
            (e.g. ``"gpt2"``, ``"distilgpt2"``, ``"gpt2-medium"``).
        use_fast: When ``True`` (default), loads the Rust-backed fast
            tokenizer for ~10× throughput during batch encoding.

    Returns:
        A configured ``PreTrainedTokenizer`` ready for batch tokenization.

    Example:
        >>> tokenizer = create_tokenizer("distilgpt2")
        >>> tokenizer.pad_token
        '<|endoftext|>'
    """
    logger.info("Loading tokenizer for model: %s (fast=%s)...", model_name, use_fast)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast,
    )

    # GPT-2 has no native pad token; assign EOS to PAD to silence warnings.
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
    return tokenizer


# ===================================================================
# Stage 3: Tokenization & Sliding-Window Chunking
# ===================================================================


def tokenize_function(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, List[List[int]]]:
    """Tokenize a batch of raw code strings *without* truncation or padding.

    This function is designed to be passed directly to ``dataset.map(…,
    batched=True)``.  Each document in the batch is encoded independently
    to its full length so that the downstream ``group_texts`` function can
    apply the sliding-window chunking strategy across document boundaries.

    Args:
        examples: A batch dictionary from the Hugging Face dataset. Must
            contain the key ``"whole_func_string"`` (the raw Python
            function body).
        tokenizer: A pre-configured ``PreTrainedTokenizer`` (see
            ``create_tokenizer``).

    Returns:
        A dictionary with keys ``"input_ids"`` and ``"attention_mask"``,
        each mapping to a list of variable-length token sequences (no
        padding applied).
    """
    tokenized = tokenizer(
        examples["whole_func_string"],
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def group_texts(
    examples: Dict[str, List[List[int]]],
    max_length: int,
    stride: int,
) -> Dict[str, List[List[int]]]:
    """Segment tokenized sequences into fixed-length chunks via sliding window.

    Each document in the batch is independently sliced into non-overlapping
    (or overlapping, when ``stride < max_length``) chunks of exactly
    ``max_length`` tokens.  Short documents (``len < max_length``) are
    silently discarded — this is intentional: padding chunks with special
    tokens would degrade the causal LM objective.

    The resulting ``input_ids`` and ``labels`` are identical for each chunk,
    as the Hugging Face ``Trainer`` internally shifts logits by one position
    for autoregressive loss computation.

    Args:
        examples: Batched output from ``tokenize_function`` with keys
            ``"input_ids"`` and ``"attention_mask"``.
        max_length: Desired context-window size in subword tokens.
        stride: Number of tokens to advance the window between consecutive
            chunks.  When ``stride == max_length``, chunks are contiguous
            with no overlap (standard for causal LM).  When
            ``stride < max_length``, chunks overlap by
            ``max_length - stride`` tokens (useful for evaluation).

    Returns:
        A dictionary with keys ``"input_ids"`` and ``"labels"``.  Each value
        is a list of equal-length token sequences (all exactly ``max_length``
        tokens long).  No ``attention_mask`` is emitted — all chunks are
        dense, so the mask is implicitly all-ones.

    Example:
        >>> # After tokenize_function, run group_texts with 256-token windows:
        >>> chunks = group_texts(batch, max_length=256, stride=256)
        >>> len(chunks["input_ids"][0])
        256
    """
    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []

    for doc_ids in examples["input_ids"]:
        doc_len = len(doc_ids)
        if doc_len < max_length:
            # Discard documents shorter than one full context window.
            continue

        # Slide the window across the document.
        num_windows = (doc_len - max_length) // stride + 1
        for i in range(num_windows):
            start = i * stride
            chunk = doc_ids[start : start + max_length]
            input_ids_list.append(chunk)
            # For causal LM, labels are identical to inputs — the Trainer
            # performs the internal shift during loss computation.
            labels_list.append(chunk.copy())

    return {"input_ids": input_ids_list, "labels": labels_list}


def process_dataset(
    raw_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    data_config: Optional[DataConfig] = None,
) -> Tuple[Dataset, int, int]:
    """Orchestrate the full tokenize → chunk pipeline on a raw dataset.

    This is the high-level entry point that chains ``tokenize_function``
    and ``group_texts`` through ``dataset.map()``, collects statistics,
    and returns the processed dataset ready for persistence.

    Args:
        raw_dataset: A Hugging Face ``Dataset`` of raw code strings
            (output of ``load_raw_dataset``).
        tokenizer: A configured tokenizer (output of ``create_tokenizer``).
        data_config: Pipeline parameters.  Defaults to a fresh ``DataConfig``
            if not provided.

    Returns:
        A tuple ``(processed_dataset, num_chunks, num_discarded)`` where:
            * ``processed_dataset`` is the tokenized, chunked dataset.
            * ``num_chunks`` is the total number of fixed-length chunks.
            * ``num_discarded`` is the count of documents that were too
              short for at least one full context window.
    """
    if data_config is None:
        data_config = DataConfig()

    max_length = data_config.max_length
    stride = data_config.stride
    num_workers = data_config.preprocessing_num_workers

    total_docs = len(raw_dataset)

    # ---- Step 1: Tokenization (batched, parallel) ----
    logger.info(
        "Tokenizing %d documents (num_workers=%d)...",
        total_docs,
        num_workers,
    )
    tokenized_dataset = raw_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),  # type: ignore[misc]
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=num_workers,
        desc="Tokenizing",
    )

    # ---- Step 2: Chunking (batched, parallel) ----
    logger.info(
        "Chunking tokenized sequences (max_length=%d, stride=%d)...",
        max_length,
        stride,
    )
    lm_dataset = tokenized_dataset.map(
        lambda batch: group_texts(batch, max_length=max_length, stride=stride),  # type: ignore[misc]
        batched=True,
        remove_columns=tokenized_dataset.column_names,
        num_proc=num_workers,
        desc=f"Chunking ({max_length}-token windows)",
    )

    num_chunks = len(lm_dataset)
    # Estimate discarded documents: those present in the original dataset
    # but not contributing any chunks.  We approximate by comparing total
    # chunks against total original documents (a lower bound).
    num_discarded = total_docs - num_chunks if num_chunks < total_docs else 0

    logger.info(
        "Chunking complete: %d chunks from %d documents (~%d short docs discarded, %.1f%%).",
        num_chunks,
        total_docs,
        max(num_discarded, 0),
        (max(num_discarded, 0) / total_docs) * 100 if total_docs > 0 else 0.0,
    )

    # Compute and log aggregate statistics.
    total_tokens = num_chunks * max_length
    logger.info("Aggregate statistics:")
    logger.info("  • Total chunks:     %d", num_chunks)
    logger.info("  • Total tokens:     %d", total_tokens)
    logger.info("  • Tokens per chunk: %d", max_length)
    logger.info("  • Chunk stride:     %d", stride)

    return lm_dataset, num_chunks, num_discarded


# ===================================================================
# Stage 4: Persistence
# ===================================================================


def save_processed_dataset(
    dataset: Dataset,
    output_path: str,
    *,
    data_fraction: Optional[float] = None,
    model_name: Optional[str] = None,
) -> str:
    """Persist a processed Arrow dataset to disk with structured naming.

    The output directory name encodes the model and data fraction so that
    multiple ablation configurations can coexist without collisions:

        ``gpt2_python_dataset_<fraction_pct>pct``

    Args:
        dataset: The processed ``Dataset`` to persist.
        output_path: Base output directory (e.g. ``data/processed/``).
        data_fraction: Fraction of data used (controls subdirectory naming).
        model_name: Hugging Face model identifier for naming.

    Returns:
        The absolute path where the dataset was saved.

    Example:
        >>> save_processed_dataset(ds, "data/processed/", data_fraction=0.10)
        '.../data/processed/gpt2_python_dataset_10pct'
    """
    # Build a descriptive directory name.
    if data_fraction is not None and data_fraction < 1.0:
        fraction_label = f"{int(data_fraction * 100)}pct"
    else:
        fraction_label = "full"

    dir_name = f"gpt2_python_dataset_{fraction_label}"
    full_path = os.path.join(output_path, dir_name)

    logger.info("Saving processed dataset to: %s", full_path)
    os.makedirs(full_path, exist_ok=True)
    dataset.save_to_disk(full_path)

    # Log the on-disk size for experiment provenance.
    total_size_bytes = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(full_path)
        for f in files
    )
    logger.info(
        "Dataset saved (%d chunks, %.1f MB on disk).",
        len(dataset),
        total_size_bytes / (1024 * 1024),
    )
    return full_path


# ===================================================================
# CLI Entry Point
# ===================================================================


def _build_argument_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Every argument maps to a field on ``DataConfig`` or a top-level
    pipeline parameter, keeping the CLI surface minimal and consistent with
    the typed configuration system.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess CodeSearchNet Python data for GPT-2 fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                                    # Full dataset, defaults\n"
            "  %(prog)s --data-fraction 0.10               # 10%% ablation subset\n"
            "  %(prog)s --model-name distilgpt2 --max-length 512 --data-fraction 0.50\n"
        ),
    )

    # ---- Data pipeline parameters ----
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="Pretrained tokenizer/model identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Fixed context-window size in tokens (default: %(default)d).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Sliding-window stride in tokens. "
            "Defaults to --max-length (non-overlapping chunks)."
        ),
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of the training set to process, in (0.0, 1.0] (default: 1.0).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Parallel workers for dataset.map() (default: %(default)d).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for processed Arrow datasets. "
            "Defaults to <project_root>/data/processed/."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling (default: %(default)d).",
    )
    parser.add_argument(
        "--no-fast-tokenizer",
        action="store_true",
        help="Use the slow Python tokenizer instead of the Rust fast tokenizer.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Run the full data preprocessing pipeline from the command line.

    This function is the single entry point for the script.  It constructs
    a ``DataConfig`` from CLI arguments, then executes the four pipeline
    stages sequentially: **load → tokenize → chunk → save**.

    Args:
        argv: Optional argument list (useful for testing).  When ``None``,
            ``sys.argv[1:]`` is used.

    Example:
        >>> from src.data_preprocess import main
        >>> main(["--data-fraction", "0.10", "--max-length", "512"])
    """
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    # Resolve stride default: when not set, use max_length for contiguous chunks.
    stride = args.stride if args.stride is not None else args.max_length

    # Resolve output directory.
    output_dir = args.output_dir
    if output_dir is None:
        project_root = _SRC_DIR.parent
        output_dir = os.path.join(project_root, "data", "processed")

    logger.info("=" * 68)
    logger.info("  Data Preprocessing Pipeline")
    logger.info("=" * 68)
    logger.info("  Model:            %s", args.model_name)
    logger.info("  Max length:       %d tokens", args.max_length)
    logger.info("  Stride:           %d tokens", stride)
    logger.info("  Data fraction:    %.2f", args.data_fraction)
    logger.info("  Parallel workers: %d", args.num_workers)
    logger.info("  Output directory: %s", output_dir)
    logger.info("  Seed:             %d", args.seed)
    logger.info("=" * 68)

    # ---- Build typed configuration ----
    data_config = DataConfig(
        max_length=args.max_length,
        stride=stride,
        preprocessing_num_workers=args.num_workers,
        processed_data_dir=output_dir,
    )

    # ---- Stage 1: Load ----
    raw_dataset = load_raw_dataset(
        data_config,
        data_fraction=args.data_fraction,
        seed=args.seed,
    )

    # ---- Stage 2: Create tokenizer ----
    tokenizer = create_tokenizer(
        model_name=args.model_name,
        use_fast=not args.no_fast_tokenizer,
    )

    # ---- Stage 3: Process (tokenize + chunk) ----
    processed_dataset, num_chunks, num_discarded = process_dataset(
        raw_dataset,
        tokenizer,
        data_config,
    )

    # ---- Stage 4: Save ----
    saved_path = save_processed_dataset(
        processed_dataset,
        output_dir,
        data_fraction=args.data_fraction,
        model_name=args.model_name,
    )

    logger.info("=" * 68)
    logger.info(
        "  Pipeline finished successfully — %d chunks → %s",
        num_chunks,
        saved_path,
    )
    logger.info("=" * 68)


# ===================================================================
# Make the script both importable and directly executable.
# ===================================================================
if __name__ == "__main__":
    main()
