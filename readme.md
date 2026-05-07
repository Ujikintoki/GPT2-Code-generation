# Domain-Adaptive Fine-Tuning of GPT-2 for Python Code Generation

This repository contains the implementation of Domain-Adaptive Fine-Tuning of OpenAI's GPT-2 (117M) specifically for semantically aware Python code generation. 

Utilizing the Python subset of the CodeSearchNet corpus and Hugging Face ecosystems, this project demonstrates how to transform a general-purpose language model into a specialized domain assistant. Rather than relying on brute-force scaling, this methodology emphasizes **strict constraints** and **learning efficiency**, exploring how effectively a standard, medium-sized causal language model can internalize the structural and syntactic priors of Python.

## Project Structure

```text
.
├── configs/                     # Resource-oriented training configurations
│   ├── high_throughput.yaml
│   └── low_vram.yaml
├── output/                      # Generated artifacts (ignored in version control)
│   ├── case_studies/            # Qualitative code generation comparisons
│   └── plots/                   # Quantitative ablation study visualizations
├── src/                         # Core library and source code
│   ├── __init__.py
│   ├── config.py                # Typed dataclasses for global hyperparameters
│   ├── data_preprocess.py       # Tokenization, whitespace preservation, and chunking
│   ├── train.py                 # Core Causal LM training and evaluation loop
│   ├── eval/                    # Evaluation modules
│   │   ├── __init__.py
│   │   ├── eval_humaneval.py    # Standardized code generation evaluation
│   │   ├── eval_model.py        # General model metric computation
│   │   └── generate_cases.py    # Qualitative case study generator
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Centralized formatting and logging utility
├── scripts/                     # Automation shell scripts
│   └── run_ablation.sh          # Orchestrates data scaling and layer ablation runs
├── requirements.txt
└── README.md
```

## Development & Execution Workflow

This project adopts a robust, hardware-agnostic workflow designed for seamless transition between local development and remote computational clusters:

- **Local Development:** Code structuring, pipeline engineering, and unit testing are conducted in a local development environment.
- **Version Control:** Changes are versioned and pushed to the repository.
- **Remote Execution:** The repository is cloned onto remote compute servers (e.g., high-performance clusters). Execution dynamically adapts to the available hardware using predefined resource profiles:
  - `low_vram` — for constrained memory environments using gradient accumulation and mixed precision.
  - `high_throughput` — for scaled environments with abundant resources.

## Setup & Installation

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

Install the core NLP and Machine Learning dependencies via the provided requirements file:

```bash
pip install -r requirements.txt
```

> **Note:** Ensure you have the appropriate CUDA-enabled PyTorch version installed for your specific hardware accelerator.

## Usage

### 1. Data Preprocessing

Process the raw CodeSearchNet Python subset into tokenized, chunked formats suitable for autoregressive training. This script preserves crucial Python whitespace semantics.

```bash
python -m src.data_preprocess
```

### 2. Model Fine-Tuning

Launch the training pipeline. You can specify the resource profile depending on your current computational environment:

```bash
# Example using a low VRAM profile via Hugging Face Accelerate
accelerate launch src/train.py --config configs/low_vram.yaml
```

### 3. Ablation Studies

To automatically run the data scaling law and capacity ablation studies across different fractions of the dataset:

```bash
bash scripts/run_ablation.sh
```

### 4. Evaluation & Case Studies

Compute quantitative metrics (e.g., Perplexity and HumanEval pass rates) and generate qualitative text comparisons:

```bash
# Generate qualitative baseline vs. fine-tuned comparisons
python -m src.eval.generate_cases

# Run structured evaluations
python -m src.eval.eval_humaneval
```

---

## Experimental Results

> **Placeholder:** Researchers should refer to the generated figures in `output/plots/` for detailed empirical findings.

- **Data Scaling & Learning Efficiency:** Visualized in `data_scaling_full.png` and `data_scaling_dual.png`.
- **Model Capacity Constraints:** Visualized in `model_capacity.png` and `model_capacity_dual.png`.
- **Architectural Ablations:** Visualized in `layer_ablation.png`.

---

## Acknowledgements

- [Hugging Face](https://huggingface.co) for providing the `transformers`, `datasets`, and `accelerate` libraries.
- [CodeSearchNet](https://github.com/github/CodeSearchNet) for the Python training corpus.