#!/usr/bin/env bash
# ==============================================================================
# Data Scaling Law Ablation Orchestration Script
# ==============================================================================
#
# This script sequentially launches ``src/train.py`` across three canonical
# data fractions — 10 %, 50 %, and 100 % — to produce the Perplexity vs.
# Dataset Size curve required for the academic report.  Each run is fully
# self-contained, writing uniquely timestamped logs to ``output/logs/`` and
# final model artifacts to ``output/final_models/``.
#
# Usage:
#   chmod +x scripts/run_ablation.sh
#   ./scripts/run_ablation.sh                          # default tier + model
#   RESOURCE_TIER=high_throughput ./scripts/run_ablation.sh
#   MODEL_NAME=gpt2-medium EPOCHS=5 ./scripts/run_ablation.sh
#
# Environment variables (all optional):
#   RESOURCE_TIER     – one of {default, low_vram, high_throughput}
#   MODEL_NAME        – HuggingFace model id  (default: gpt2)
#   DATA_PATH         – path to preprocessed Arrow dataset dir
#   EPOCHS            – number of training epochs
#   LEARNING_RATE     – peak learning rate
#   SEED              – global random seed
#   EXTRA_TRAIN_ARGS  – any additional flags forwarded to src/train.py
# ==============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Resolve the project root (the directory containing this script's parent)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# 2. User-configurable knobs (environment variable overrides)
# ---------------------------------------------------------------------------
RESOURCE_TIER="${RESOURCE_TIER:-low_vram}"         # safe default ≤16 GB VRAM
MODEL_NAME="${MODEL_NAME:-gpt2}"                   # baseline GPT-2 124M
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
SEED="${SEED:-42}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/processed/gpt2_python_dataset_full}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

# Data fractions to iterate over for the scaling-law curve.
DATA_FRACTIONS=(0.10 0.50 1.0)

# ---------------------------------------------------------------------------
# 3. Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -d "${DATA_PATH}" ]; then
    echo "✗  Preprocessed dataset not found at: ${DATA_PATH}"
    echo "   Run 'python -m src.data_preprocess --data-fraction 1.0' first."
    exit 1
fi

echo "================================================================================"
echo "  Data Scaling Law Ablation Study"
echo "================================================================================"
echo "  Project root:    ${PROJECT_ROOT}"
echo "  Data path:       ${DATA_PATH}"
echo "  Model:           ${MODEL_NAME}"
echo "  Resource tier:   ${RESOURCE_TIER}"
echo "  Epochs:          ${EPOCHS}"
echo "  Learning rate:   ${LEARNING_RATE}"
echo "  Seed:            ${SEED}"
echo "  Fractions:       ${DATA_FRACTIONS[*]}"
echo "================================================================================"
echo ""

# Create output directories if they do not exist.
mkdir -p output/logs output/checkpoints output/final_models

# ---------------------------------------------------------------------------
# 4. Run training sequentially for each data fraction
# ---------------------------------------------------------------------------
for frac in "${DATA_FRACTIONS[@]}"; do
    FRAC_PCT=$(awk "BEGIN {printf \"%d\", ${frac} * 100}")
    EXP_NAME="ablation-${MODEL_NAME}-${FRAC_PCT}pct"

    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "  ▶  Launching ablation run: ${EXP_NAME}"
    echo "     Data fraction: ${frac}  (${FRAC_PCT} %)"
    echo "--------------------------------------------------------------------------------"

    python -m src.train \
        --data_path "${DATA_PATH}" \
        --model_name "${MODEL_NAME}" \
        --resource_tier "${RESOURCE_TIER}" \
        --epochs "${EPOCHS}" \
        --learning_rate "${LEARNING_RATE}" \
        --seed "${SEED}" \
        --data_fraction "${frac}" \
        --experiment_name "${EXP_NAME}" \
        ${EXTRA_TRAIN_ARGS}

    echo ""
    echo "  ✔  Completed: ${EXP_NAME}"
    echo "     Logs →  output/logs/${EXP_NAME}_*.log"
    echo "     Model → output/final_models/"
done

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "  Ablation study complete — all ${#DATA_FRACTIONS[@]} fractions finished."
echo ""
echo "  Logs:      output/logs/"
echo "  Models:    output/final_models/"
echo ""
echo "  Next step: Generate the qualitative case studies with"
echo "     python -m src.eval.generate_cases"
echo "================================================================================"