"""Qualitative case-study generator for fine-tuned vs. baseline GPT-2 models.

This module loads the baseline (zero-shot) GPT-2 and a domain-adapted
fine-tuned checkpoint side by side, feeds both identical Python coding
prompts, and collects their autoregressive completions.  The comparative
output is rendered as a structured Markdown report suitable for direct
inclusion in an academic paper.

Architecture
------------
1. **Prompt Bank** — A curated set of Python function signatures,
   docstrings, and partial code snippets covering diverse coding tasks
   (list manipulation, recursion, OOP, generators, error handling, etc.).
2. **Model Loading** — Loads the *baseline* model (the original pretrained
   GPT-2 from Hugging Face Hub) and a *fine-tuned* model (from a saved
   checkpoint directory) using the same tokenizer for both.
3. **Generation** — Autoregressive greedy/constrained-sampling decoding
   with code-appropriate hyperparameters (low temperature, top-p
   nucleation, repetition penalty).
4. **Rendering** — Produces a single Markdown file
   (``output/case_studies.md``) with side-by-side code blocks labelled
   "Baseline (Zero-Shot)" and "Fine-Tuned (Domain-Adapted)".

Usage
-----
    # Default paths:
    python -m src.eval.generate_cases

    # Custom model paths:
    python -m src.eval.generate_cases \\
        --baseline_model gpt2 \\
        --fine_tuned_path ./output/final_models/gpt2-code-best \\
        --output_path ./output/case_studies.md

    # Use curated prompt file:
    python -m src.eval.generate_cases --prompts_file ./configs/case_prompts.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (  # type: ignore[import-not-found]
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# ---------------------------------------------------------------------------
# Ensure ``src/`` is on the Python path when executed as a script so that
# config and logger imports work regardless of invocation directory.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.logger import setup_logger  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = setup_logger(__name__, experiment_name="case-study-generation")


# ===================================================================
# Curated Python Prompt Bank
# ===================================================================
# Each prompt is a realistic, self-contained coding task that exercises
# different aspects of Python (algorithms, data structures, OOP, generators,
# error handling, etc.).  The prompts are deliberately left *incomplete* so
# that the model must synthesise the body.
# ===================================================================

_DEFAULT_PROMPTS: List[str] = [
    # --- List / sequence manipulation ---
    'def fibonacci(n: int) -> list[int]:\n    """Return the first n Fibonacci numbers."""\n',
    'def remove_duplicates(items: list) -> list:\n    """Return a new list with duplicates removed, preserving order."""\n',
    'def binary_search(arr: list[int], target: int) -> int:\n    """Return index of target in sorted arr, or -1 if not found."""\n',

    # --- String processing ---
    'def reverse_words(sentence: str) -> str:\n    """Reverse the order of words in the sentence."""\n',
    'def is_palindrome(s: str) -> bool:\n    """Check whether s reads the same forwards and backwards, ignoring case and punctuation."""\n',

    # --- Recursion ---
    'def factorial(n: int) -> int:\n    """Compute n! recursively."""\n',
    'def flatten(nested: list) -> list:\n    """Flatten a nested list of arbitrary depth into a flat list."""\n',

    # --- OOP ---
    'class Counter:\n    """A simple counter that can be incremented, decremented, and reset."""\n\n    def __init__(self, start: int = 0) -> None:\n',
    'class Stack:\n    """A LIFO stack with push, pop, and peek operations."""\n\n    def __init__(self) -> None:\n',

    # --- Generators ---
    'def read_lines(filename: str) -> list[str]:\n    """Read all lines from a file and return them as a list, stripping whitespace."""\n',
    'def chunked(iterable, size: int):\n    """Yield successive chunks of the given size from the iterable."""\n',

    # --- Error handling ---
    'def safe_divide(a: float, b: float) -> float:\n    """Divide a by b, returning 0.0 if b is zero."""\n',

    # --- Functional / comprehensions ---
    'def square_even(numbers: list[int]) -> list[int]:\n    """Square every even number in the list."""\n',
    'def group_by_first_letter(words: list[str]) -> dict[str, list[str]]:\n    """Group words by their first letter."""\n',

    # --- File I/O ---
    'def count_lines_in_file(filepath: str) -> int:\n    """Count the number of lines in a text file."""\n',
]


# ===================================================================
# Dataclass for generation configuration
# ===================================================================


@dataclass(frozen=True)
class GenerationConfig:
    """Hyperparameters controlling autoregressive text generation.

    These defaults are tuned for **code generation** tasks: low
    temperature reduces randomness, top-p nucleation prunes the tail
    of the distribution, and a repetition penalty discourages stuttering.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate beyond the
            prompt.
        temperature: Softmax temperature; lower = more deterministic.
        top_p: Cumulative probability threshold for nucleus sampling.
        top_k: Keep only the top_k most probable tokens (0 = disabled).
        do_sample: When ``False``, greedy decoding is used (temperature=1).
        repetition_penalty: Penalty factor for repeated n-grams (>1.0).
        pad_token_id: Token ID used for padding.
        eos_token_id: Token ID that signals end of generation.
    """

    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 0
    do_sample: bool = False  # Greedy for reproducibility
    repetition_penalty: float = 1.0
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


# ===================================================================
# Helper: model loading
# ===================================================================


def _load_model_and_tokenizer(
    model_path: str,
    device: torch.device,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a causal LM and its tokenizer from a local path or HF Hub id.

    Args:
        model_path: Either a Hugging Face Hub identifier (e.g. ``"gpt2"``)
            or a local directory containing a saved model + tokenizer.
        device: The torch device to move the model onto.

    Returns:
        A tuple ``(model, tokenizer)`` with the model in evaluation mode.

    Raises:
        FileNotFoundError: If ``model_path`` is a local directory that does
            not exist or does not contain a valid model.
        OSError: If the Hugging Face Hub model cannot be downloaded.
    """
    logger.info("Loading tokenizer from: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # GPT-2 has no pad token by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Assigned eos_token as pad_token.")

    logger.info("Loading model from: %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %d parameters on %s.", total_params, device)
    return model, tokenizer


# ===================================================================
# Helper: generation
# ===================================================================


def generate_completion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    gen_cfg: GenerationConfig,
    device: torch.device,
) -> str:
    """Run autoregressive generation from a prompt and return the completion.

    The function tokenizes the prompt, calls ``model.generate()`` with
    the configured hyperparameters, and decodes only the new tokens.

    Args:
        model: A causal LM in evaluation mode.
        tokenizer: The tokenizer matching the model.
        prompt: The input code snippet (function signature + docstring).
        gen_cfg: Frozen ``GenerationConfig`` with sampling hyperparameters.
        device: The torch device holding the model.

    Returns:
        The generated continuation as a string (prompt + completion).
        Leading/trailing whitespace is stripped.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature if gen_cfg.do_sample else 1.0,
            top_p=gen_cfg.top_p if gen_cfg.do_sample else 1.0,
            top_k=gen_cfg.top_k if gen_cfg.do_sample else 0,
            do_sample=gen_cfg.do_sample,
            repetition_penalty=gen_cfg.repetition_penalty,
            pad_token_id=gen_cfg.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=gen_cfg.eos_token_id or tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens.
    prompt_length = inputs.input_ids.shape[1]
    generated_ids = output_ids[0][prompt_length:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return prompt + completion


def _truncate_at_block_boundary(text: str, prompt: str) -> str:
    """Truncate generated text at the first blank line or class/function def.

    This prevents the model from rambling into unrelated code after it has
    completed the requested function body.

    Args:
        text: The full generated text (prompt + completion).
        prompt: The original prompt to exclude from truncation logic.

    Returns:
        The text truncated at a sensible stopping point, with trailing
        whitespace removed.
    """
    completion_only = text[len(prompt):]
    lines = completion_only.split("\n")

    # Stop at the first blank line (paragraph break).
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Stop at a new top-level definition or blank line
        if stripped == "":
            return prompt + "\n".join(lines[:i]) + "\n"
        if (stripped.startswith("def ") or stripped.startswith("class ") or
                stripped.startswith("if __name__")):
            if i > 1:  # Allow at least one line of body
                return prompt + "\n".join(lines[:i]) + "\n"

    return text


# ===================================================================
# Markdown Report Renderer
# ===================================================================


def _escape_markdown_code(text: str) -> str:
    """Escape backticks inside inline code for safe Markdown embedding.

    Args:
        text: Raw text that may contain backtick sequences.

    Returns:
        Text with triple-backtick sequences escaped.
    """
    # If the text already contains triple backticks, use quadruple backticks
    # to fence the code block (CommonMark spec).
    if "```" in text:
        return text.replace("```", "````")
    return text


def render_case_study_markdown(
    cases: List[Dict[str, str]],
    output_path: str,
    *,
    baseline_label: str = "GPT-2 (Zero-Shot)",
    fine_tuned_label: str = "GPT-2 (Fine-Tuned)",
    title: str = "Qualitative Case Study: Code Generation Comparison",
) -> None:
    """Render a list of comparative cases into a structured Markdown report.

    Args:
        cases: A list of dictionaries, each with keys:
            * ``"prompt"`` — the input prompt.
            * ``"baseline"`` — the baseline model completion.
            * ``"fine_tuned"`` — the fine-tuned model completion.
        output_path: Absolute or relative path for the output ``.md`` file.
        baseline_label: Label for the baseline column header.
        fine_tuned_label: Label for the fine-tuned column header.
        title: Top-level heading for the report.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lines: List[str] = []
    lines.append(f"# {title}\n")
    lines.append(
        f"> Generated on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    lines.append(
        "> Each case shows the identical prompt fed to both models, "
        "followed by side-by-side completions.\n"
    )
    lines.append("---\n")

    for idx, case in enumerate(cases, start=1):
        prompt = case["prompt"]
        baseline = case["baseline"]
        fine_tuned = case["fine_tuned"]

        lines.append(f"## Case {idx}\n")
        lines.append("### Prompt\n")
        lines.append("```python")
        lines.append(prompt.rstrip())
        lines.append("```\n")

        lines.append(f"### {baseline_label}\n")
        lines.append("```python")
        lines.append(_escape_markdown_code(baseline.rstrip()))
        lines.append("```\n")

        lines.append(f"### {fine_tuned_label}\n")
        lines.append("```python")
        lines.append(_escape_markdown_code(fine_tuned.rstrip()))
        lines.append("```\n")

        lines.append("---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Case study report written to: %s (%d cases)", output_path, len(cases))


# ===================================================================
# High-Level Orchestration
# ===================================================================


def run_case_study(
    baseline_model_name: str,
    fine_tuned_path: str,
    prompts: List[str],
    *,
    gen_cfg: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
) -> List[Dict[str, str]]:
    """Execute the side-by-side case study across all prompts.

    Args:
        baseline_model_name: Hugging Face Hub id for the zero-shot model.
        fine_tuned_path: Path to the fine-tuned checkpoint directory.
        prompts: List of Python prompt strings.
        gen_cfg: Generation hyperparameters.  Uses code-tuned defaults
            when ``None``.
        device: Torch device.  Auto-detected when ``None``.

    Returns:
        A list of dictionaries with keys ``"prompt"``, ``"baseline"``,
        and ``"fine_tuned"``.

    Raises:
        FileNotFoundError: If the fine-tuned checkpoint cannot be found.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gen_cfg is None:
        gen_cfg = GenerationConfig()

    logger.info("Device: %s", device)

    # ---- Load baseline model ----
    logger.info("Loading baseline (zero-shot) model: %s", baseline_model_name)
    baseline_model, baseline_tokenizer = _load_model_and_tokenizer(
        baseline_model_name, device
    )
    baseline_gen_cfg = GenerationConfig(
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        top_k=gen_cfg.top_k,
        do_sample=gen_cfg.do_sample,
        repetition_penalty=gen_cfg.repetition_penalty,
        pad_token_id=baseline_tokenizer.eos_token_id,
        eos_token_id=baseline_tokenizer.eos_token_id,
    )

    # ---- Load fine-tuned model ----
    logger.info("Loading fine-tuned model: %s", fine_tuned_path)
    if not os.path.isdir(fine_tuned_path):
        raise FileNotFoundError(
            f"Fine-tuned model directory not found: {fine_tuned_path}. "
            "Run src/train.py first to produce a checkpoint."
        )
    finetuned_model, finetuned_tokenizer = _load_model_and_tokenizer(
        fine_tuned_path, device
    )
    finetuned_gen_cfg = GenerationConfig(
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        top_k=gen_cfg.top_k,
        do_sample=gen_cfg.do_sample,
        repetition_penalty=gen_cfg.repetition_penalty,
        pad_token_id=finetuned_tokenizer.eos_token_id,
        eos_token_id=finetuned_tokenizer.eos_token_id,
    )

    # ---- Generate for each prompt ----
    cases: List[Dict[str, str]] = []

    for i, prompt in enumerate(prompts):
        logger.info(
            "Generating case %d/%d — prompt starts with: %s",
            i + 1,
            len(prompts),
            prompt[:60].replace("\n", "\\n"),
        )

        baseline_output = generate_completion(
            baseline_model, baseline_tokenizer, prompt, baseline_gen_cfg, device
        )
        baseline_output = _truncate_at_block_boundary(baseline_output, prompt)

        finetuned_output = generate_completion(
            finetuned_model, finetuned_tokenizer, prompt, finetuned_gen_cfg, device
        )
        finetuned_output = _truncate_at_block_boundary(finetuned_output, prompt)

        cases.append(
            {
                "prompt": prompt,
                "baseline": baseline_output,
                "fine_tuned": finetuned_output,
            }
        )

    logger.info("All %d cases generated successfully.", len(cases))
    return cases


# ===================================================================
# CLI Entry Point
# ===================================================================


def _build_argument_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the case study generator."""
    parser = argparse.ArgumentParser(
        description="Generate qualitative case studies comparing baseline vs. fine-tuned GPT-2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --fine_tuned_path ./output/final_models/gpt2-code --baseline_model gpt2-medium\n"
            "  %(prog)s --prompts_file ./configs/my_prompts.json --max_new_tokens 256\n"
        ),
    )

    parser.add_argument(
        "--baseline_model",
        type=str,
        default="gpt2",
        help="Hugging Face Hub id for the zero-shot baseline model (default: %(default)s).",
    )
    parser.add_argument(
        "--fine_tuned_path",
        type=str,
        default=None,
        help=(
            "Path to the fine-tuned model checkpoint directory. "
            "Defaults to <project_root>/output/final_models/"
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Path for the output Markdown report. "
            "Defaults to <project_root>/output/case_studies.md"
        ),
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing a list of prompt strings. "
            "When not provided, a curated default prompt bank is used."
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per completion (default: %(default)d).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Softmax temperature; lower = more deterministic (default: %.1f).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability threshold (default: %.2f).",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable stochastic sampling (default: greedy decoding).",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty >1.0 discourages token repeats (default: %.1f).",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI arguments, generate case studies, and write the Markdown report.

    Args:
        argv: Optional argument list (useful for testing).
    """
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    # Resolve default paths relative to project root.
    project_root = _SRC_DIR.parent

    fine_tuned_path: str = args.fine_tuned_path
    if fine_tuned_path is None:
        fine_tuned_path = os.path.join(project_root, "output", "final_models")

    output_path: str = args.output_path
    if output_path is None:
        output_path = os.path.join(project_root, "output", "case_studies.md")

    # Load prompts.
    if args.prompts_file is not None:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts: List[str] = json.load(f)
        logger.info("Loaded %d prompts from: %s", len(prompts), args.prompts_file)
    else:
        prompts = _DEFAULT_PROMPTS
        logger.info("Using default prompt bank (%d prompts).", len(prompts))

    # Build generation config.
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
    )

    logger.info("=" * 68)
    logger.info("  Qualitative Case Study Generator")
    logger.info("=" * 68)
    logger.info("  Baseline model:    %s", args.baseline_model)
    logger.info("  Fine-tuned path:   %s", fine_tuned_path)
    logger.info("  Output path:       %s", output_path)
    logger.info("  Prompts:           %d", len(prompts))
    logger.info("  Max new tokens:    %d", gen_cfg.max_new_tokens)
    logger.info("  Temperature:       %.3f", gen_cfg.temperature)
    logger.info("  Top-p:             %.3f", gen_cfg.top_p)
    logger.info("  Sampling:          %s", "on" if gen_cfg.do_sample else "off (greedy)")
    logger.info("=" * 68)

    # ---- Generate cases ----
    cases = run_case_study(
        baseline_model_name=args.baseline_model,
        fine_tuned_path=fine_tuned_path,
        prompts=prompts,
        gen_cfg=gen_cfg,
    )

    # ---- Render Markdown ----
    render_case_study_markdown(
        cases,
        output_path,
        baseline_label=f"{args.baseline_model} (Zero-Shot)",
        fine_tuned_label="Fine-Tuned (Domain-Adapted)",
    )

    logger.info("Done. Report saved to: %s", output_path)


# ===================================================================
# Make the script both importable and directly executable.
# ===================================================================
if __name__ == "__main__":
    main()