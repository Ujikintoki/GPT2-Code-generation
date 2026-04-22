"""
Data Preprocessing for Domain-Adaptive Fine-Tuning.

This script tokenizes and chunks raw Python code data using a sliding window
(or sequential chunking) strategy, preparing it for GPT-2 causal language modeling.
"""

import logging
import os
from typing import Any, Dict, List

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

# Configure standard academic logging format
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ================= Configuration =================
MODEL_CHECKPOINT: str = "gpt2"
MAX_LENGTH: int = 256  # The fixed context window size for GPT-2
STRIDE: int = 256  # Stride = MAX_LENGTH indicates sequential chunking with no overlap

OUTPUT_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
)


def tokenize_and_chunk(
    examples: Dict[str, List[Any]], tokenizer: PreTrainedTokenizer
) -> Dict[str, List[List[int]]]:
    """
    Tokenize raw code strings and apply chunking.

    Extracts chunks of exactly MAX_LENGTH tokens. Chunks are generated
    based on the configured STRIDE parameter.
    """
    tokenized_inputs = tokenizer(
        examples["whole_func_string"], add_special_tokens=False
    )

    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []

    for input_ids in tokenized_inputs["input_ids"]:
        if len(input_ids) >= MAX_LENGTH:
            for i in range(0, len(input_ids) - MAX_LENGTH + 1, STRIDE):
                chunk = input_ids[i : i + MAX_LENGTH]
                input_ids_list.append(chunk)
                labels_list.append(chunk.copy())

    return {"input_ids": input_ids_list, "labels": labels_list}


def main() -> None:
    logger.info("Initializing data preprocessing pipeline.")

    # 1. Load Dataset
    logger.info("Loading CodeSearchNet (Python subset) full training split...")
    raw_datasets = load_dataset("code_search_net", "python", split="train")

    # 2. Load Tokenizer
    logger.info("Loading GPT-2 BPE Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Process Dataset
    logger.info(
        "Executing Tokenization and Chunking (MAX_LENGTH=%d, STRIDE=%d)...",
        MAX_LENGTH,
        STRIDE,
    )
    lm_datasets = raw_datasets.map(
        lambda examples: tokenize_and_chunk(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets.column_names,
        desc="Running sequential chunking",
    )

    logger.info(
        "Processing complete! Transformed raw documents into %d chunks of length %d.",
        len(lm_datasets),
        MAX_LENGTH,
    )

    # 4. Save to Disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "gpt2_python_dataset")

    logger.info("Saving processed data to local disk: %s", save_path)
    lm_datasets.save_to_disk(save_path)
    logger.info("Data preprocessing finished successfully.")


if __name__ == "__main__":
    main()
