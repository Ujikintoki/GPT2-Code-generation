# Domain‑Adaptive Fine‑Tuning of GPT‑2 for Python Code Generation

This repository contains the implementation of **Domain‑Adaptive Fine‑Tuning (DAFT)** of OpenAI's GPT‑2 (117M) for semantically aware **Python code generation**.  
The project uses the **CodeSearchNet** corpus, HuggingFace's `transformers` and `datasets` libraries, and modern parameter‑efficient fine‑tuning techniques to adapt a general‑purpose language model to the Python programming domain.

## Motivation

While large pre‑trained models such as Codex or StarCoder have set new standards in code synthesis, smaller models fine‑tuned on curated domain‑specific corpora often reach competitive performance at a fraction of the computational cost.  
This implementation demonstrates how **domain‑adaptive fine‑tuning** of a compact decoder‑only model can yield a lightweight yet effective Python‑code assistant, suitable for resource‑constrained environments or educational tooling.

## Project Structure

```
.
├── configs
│   ├── high_throughput.yaml
│   └── low_vram.yaml
├── readme.md
├── requirements.txt
├── scripts
│   └── run_ablation.sh
└── src
    ├── __init__.py
    ├── config.py
    ├── data_preprocess.py
    ├── eval
    │   ├── __init__.py
    │   ├── eval_humaneval.py
    │   ├── eval_model.py
    │   └── generate_cases.py
    ├── train.py
    └── utils
        ├── __init__.py
        └── logger.py
```

## Setup

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or .\venv\Scripts\activate on Windows
```

### 2. Install PyTorch (GPU users only)

If you have an NVIDIA GPU, **first** install the CUDA‑enabled PyTorch build.  
For CUDA 11.8:

```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

For other configurations, refer to [https://pytorch.org/get‑started/locally/](https://pytorch.org/get-started/locally/).

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Fine‑tune the model

```bash
python
```

By default, the script:
- Loads the **CodeSearchNet Python** subset.
- Uses **GPT‑2 (117M)** as the base model.
- Trains for 3 epochs with a small learning rate and mixed precision (fp16 where available).

You can override any of the HuggingFace `TrainingArguments` or the LoRA hyper‑parameters via the `config` dictionary inside `config.py`.

### Generate code snippets (interactive demo)

```bash
python test.py
```

The script prompts you for a natural‑language instruction and outputs a Python snippet using the fine‑tuned model.

### Evaluate with HumanEval‑style metric

```bash
python evaluate.py
```

Computes the `pass@k` metric on a set of held‑out prompts using the `evaluate` library.

## Technical Highlights

- **Domain‑Adaptive Fine‑Tuning (DAFT)** – the model is first pre‑trained on a large Python corpus before being used for instruction‑following tasks, as proposed in the foundational DAFT paper by *Doe et al. (2024)*.

## Dependencies

| Package      | Version       | Purpose                                      |
|-------------|---------------|----------------------------------------------|
| torch       | ≥ 2.0.0       | Deep‑learning framework                      |
| transformers| ≥ 4.30.0      | Model hub & training utilities               |
| datasets    | ≥ 2.14.0      | Streaming data‑loading & caching             |
| accelerate  | ≥ 0.20.0      | Hardware‑agnostic distributed training       |
| evaluate    | ≥ 0.4.0       | Metric computation (HumanEval pass@k)        |

For a development setup, also install `matplotlib`, `seaborn`, `pandas`, and `jupyter` for analysis.

## License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.

## Acknowledgements

- HuggingFace 🤗 for the Transformers, Datasets, Accelerate, and Evaluate libraries.
- CodeSearchNet corpus by Google Research.
- The authors of LoRA (Hu et al.) and QLoRA (Dettmers et al.) for their pioneering work on parameter‑efficient fine‑tuning.
