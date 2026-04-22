import os
import torch
import math
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= Configuration =================
MODEL_CHECKPOINT = "gpt2"
# Dynamically locate the data processed in the previous step
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "debug_sample")

def main():
    # 1. Setup Device (Support for Apple Silicon MPS, NVIDIA CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS) for acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA) for acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU. Evaluation might be slow.")

    # 2. Load the Pre-trained Model and Tokenizer (Zero-shot, NO fine-tuning)
    print(f"\nLoading baseline model '{MODEL_CHECKPOINT}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the base Causal LM model
    model = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINT)
    model.to(device)
    model.eval() # Set model to evaluation mode (disables dropout layers, etc.)

    # ================= PART 1: Quantitative Evaluation (Perplexity) =================
    print(f"\n--- Part 1: Quantitative Evaluation (Perplexity) ---")
    try:
        dataset = load_from_disk(DATA_PATH)
        print(f"Loaded debug dataset with {len(dataset)} chunks.")
        
        total_loss = 0.0
        total_batches = len(dataset)
        
        # Disable gradient calculation to save memory and speed up evaluation
        with torch.no_grad():
            for i in range(total_batches):
                # Convert data lists to PyTorch tensors and move to device
                input_ids = torch.tensor([dataset[i]["input_ids"]]).to(device)
                labels = torch.tensor([dataset[i]["labels"]]).to(device)
                
                # Forward pass: the model automatically calculates Cross Entropy Loss when labels are provided
                outputs = model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
        
        # Calculate Perplexity (PPL)
        avg_loss = total_loss / total_batches
        perplexity = math.exp(avg_loss)
        print(f"📊 Baseline Average Loss: {avg_loss:.4f}")
        print(f"📊 Baseline Perplexity (PPL): {perplexity:.4f}")
        
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}. Please run data_preprocess.py first.")

    # ================= PART 2: Qualitative Evaluation (Case Studies) =================
    print(f"\n--- Part 2: Qualitative Evaluation (Case Studies) ---")
    # Define a few standard Python prompts
    prompts = [
        "def calculate_average(numbers):",
        "import requests\n\ndef fetch_webpage(url):"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n[Test Case {i+1}] Prompt:\n{prompt}")
        
        # Tokenize the input string
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate code continuation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=60,         # Maximum tokens to generate
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,        # Use sampling instead of greedy search
                temperature=0.7,       # Control randomness (0.7 is a standard balance)
                top_k=50               # Nucleus sampling parameter
            )
        
        # Decode the generated tensor back to human-readable text
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("-" * 30)
        print(generated_code)
        print("-" * 30)

if __name__ == "__main__":
    main()