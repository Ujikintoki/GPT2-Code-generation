"""
Unified HumanEval Evaluation Script for Domain-Adaptive Models.

This script manages both the autoregressive generation of Python code
and the subsequent functional execution (pass@k computation) via a sandbox.
It dynamically adapts inference precision and batching strategies based on
the target hardware architecture (T4 vs. A100).
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any

import torch
from datasets import load_dataset
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

# Essential for the evaluate library to safely execute generated Python code
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# Configure standard academic logging format
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for unified HumanEval evaluation."""
    parser = argparse.ArgumentParser(description="Unified HumanEval Evaluation")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./models/eval_temp/humaneval_samples.jsonl",
        help="Path to save the intermediate JSONL generation results.",
    )
    parser.add_argument(
        "--gpu_target",
        type=str,
        default="T4",
        choices=["T4", "A100"],
        help="Target hardware architecture for dynamic inference routing.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate per problem (n for pass@k).",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 10],
        help="List of k values for pass@k metric (e.g., --k_values 1 10).",
    )
    return parser.parse_args()


def configure_inference_profile(gpu_target: str) -> Dict[str, Any]:
    """
    Dynamically route hardware-specific inference arguments.

    Args:
        gpu_target: String identifier of the hardware.

    Returns:
        Dict specifying target device and torch dtype.
    """
    profile = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}

    if gpu_target == "A100" and torch.cuda.is_bf16_supported():
        logger.info("Applying A100 inference profile: bfloat16 precision.")
        profile["dtype"] = torch.bfloat16
    else:
        logger.info("Applying T4 inference profile: float16 precision.")
        profile["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    return profile


def main() -> None:
    args = parse_args()
    hw_profile = configure_inference_profile(args.gpu_target)
    device = hw_profile["device"]

    logger.info("Initiating Phase 1: Safe Generation for model: %s", args.model_path)

    # ==========================================
    # Phase 1: Safe Generation (Generation)
    # ==========================================
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=hw_profile["dtype"]
        ).to(device)
        model.eval()
    except Exception as e:
        logger.error("Failed to load model from %s. Error: %s", args.model_path, str(e))
        sys.exit(1)

    logger.info("Loading openai_humaneval via HF Datasets (bypassing git clone).")
    try:
        dataset = load_dataset("openai_humaneval", split="test")
    except Exception as e:
        logger.error("Dataset load failure. Error: %s", str(e))
        sys.exit(1)

    results = []

    logger.info("Commencing autoregressive generation...")
    with torch.no_grad():
        for i, task in enumerate(dataset):
            task_id = task["task_id"]
            prompt = task["prompt"]

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Constrained generation parameters to prevent hallucination overflow
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.8 if args.num_samples > 1 else 0.2,
                do_sample=True if args.num_samples > 1 else False,
                top_p=0.95,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.eos_token_id,
            )

            for j in range(args.num_samples):
                gen_tokens = outputs[j][inputs.input_ids.shape[1] :]
                generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                # Structural truncation mapping
                completion = (
                    generated_text.split("\ndef ")[0]
                    .split("\nclass ")[0]
                    .split("\nif __name__")[0]
                )

                results.append({"task_id": task_id, "completion": completion})

            if (i + 1) % 20 == 0 or (i + 1) == len(dataset):
                logger.info(
                    "Generation progress: %d/%d tasks completed.", i + 1, len(dataset)
                )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    logger.info("Phase 1 Complete. Checkpoints saved to %s", args.output_path)

    # ==========================================
    # Phase 2: Restricted Execution (Evaluation)
    # ==========================================
    logger.info("Initiating Phase 2: Restricted Execution and pass@k Evaluation")
    code_eval = evaluate.load("code_eval")

    test_cases = [task["test"] for task in dataset]
    predictions = [[] for _ in range(len(dataset))]
    task_id_to_idx = {task["task_id"]: idx for idx, task in enumerate(dataset)}

    for res in results:
        idx = task_id_to_idx[res["task_id"]]
        predictions[idx].append(res["completion"])

    logger.info("Executing unit tests in HF evaluate sandbox...")
    pass_at_k, _ = code_eval.compute(
        references=test_cases, predictions=predictions, k=args.k_values
    )

    print("\n")
    print("=" * 60)
    print(f"{'HumanEval Functional Metric':<30} | {'Value':<25}")
    print("-" * 60)
    for k_val in args.k_values:
        metric_key = f"pass@{k_val}"
        if metric_key in pass_at_k:
            print(f"{metric_key:<30} | {pass_at_k[metric_key]:<25.4f}")
    print("=" * 60)
    print("\n")


if __name__ == "__main__":
    main()
