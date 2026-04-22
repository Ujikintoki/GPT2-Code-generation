import os
from datasets import load_dataset
from transformers import AutoTokenizer

# ================= Configuration =================
MODEL_CHECKPOINT = "gpt2"
MAX_LENGTH = 256  # The fixed context window size for GPT-2
STRIDE = 256  # The sliding window stride to overlap data
# Define local output directory (e.g., ../data/processed)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "processed"
)


def tokenize_and_chunk(examples, tokenizer):
    """
    Tokenize the raw code strings and apply a sliding window chunking strategy.
    Unlike naive concatenation, this method isolates each document and extracts
    overlapping chunks to preserve structural coherence.
    """
    # 1. Tokenize the entire batch of code strings without truncation
    # We do not use padding here because we will manually chunk them to MAX_LENGTH
    tokenized_inputs = tokenizer(
        examples["whole_func_string"], add_special_tokens=False
    )

    input_ids_list = []
    labels_list = []

    # 2. Apply sliding window chunking for each document in the batch
    for input_ids in tokenized_inputs["input_ids"]:
        # Only process documents that are longer than the required MAX_LENGTH
        if len(input_ids) >= MAX_LENGTH:
            # Slide the window across the document with the specified STRIDE
            for i in range(0, len(input_ids) - MAX_LENGTH + 1, STRIDE):
                # Extract a chunk of exactly MAX_LENGTH tokens
                chunk = input_ids[i : i + MAX_LENGTH]

                input_ids_list.append(chunk)

                # For Causal Language Modeling (CLM), the labels are identical to the input_ids.
                # The Hugging Face Trainer will automatically shift the labels internally.
                labels_list.append(chunk.copy())

    # Note: We omit the 'attention_mask' because all chunks are exactly MAX_LENGTH
    # (no padding tokens are present), which is optimal for Causal LM training.
    return {"input_ids": input_ids_list, "labels": labels_list}


def main():
    print("1. Loading CodeSearchNet (Python subset) debug sample...")
    # Using a 1% split for local debugging. Change to split="train" for cloud execution.
    # raw_datasets = load_dataset("code_search_net", "python", split="train[:1%]")
    raw_datasets = load_dataset("code_search_net", "python", split="train")

    print("2. Loading GPT-2 BPE Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    # Assign EOS token to PAD token to avoid warning messages, though we don't strictly use padding here
    tokenizer.pad_token = tokenizer.eos_token

    print("3. Executing Tokenization and Sliding Window Chunking...")
    # Apply the processing function to the dataset in batches
    # remove_columns deletes the raw text strings to save memory, keeping only the tensors
    lm_datasets = raw_datasets.map(
        lambda examples: tokenize_and_chunk(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets.column_names,
        desc="Running sliding window chunking",
    )

    print(
        f"Processing complete! {len(raw_datasets)} raw documents transformed into {len(lm_datasets)} chunks of length {MAX_LENGTH}."
    )

    print("4. Saving processed debug data to local disk...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "debug_sample")
    lm_datasets.save_to_disk(save_path)
    print(f"Data successfully saved to: {save_path}")


if __name__ == "__main__":
    main()
