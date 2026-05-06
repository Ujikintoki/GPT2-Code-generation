"""
Plot generation script for Project Results (Dual Y-Axis Version).

This script utilizes Matplotlib, Seaborn, and Numpy to generate academic-grade
visualizations for the Domain-Adaptive Fine-Tuning experiments. It plots both
Perplexity (PPL) and Cross-Entropy Loss simultaneously using twin axes.
"""

import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configure logging per project standards
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def configure_academic_style() -> None:
    """
    Configure standard academic plotting styles.
    """
    logger.info("Configuring academic plotting styles.")
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_model_capacity(output_dir: str) -> None:
    """
    Generate a grouped bar chart comparing model capacity with dual axes for PPL and Loss.
    """
    logger.info("Generating model capacity plot (Dual Axis).")
    models: List[str] = [
        "DistilGPT-2\n(82M)",
        "GPT-2 Base\n(117M)",
        "GPT-2 Medium\n(345M)",
    ]
    ppl_ft: List[float] = [4.42, 3.81, 3.34]
    loss_ft: List[float] = [1.4860, 1.3365, 1.2051]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    # Plot bars
    bars1 = ax1.bar(
        x - width / 2, ppl_ft, width, color="#4C72B0", label="Perplexity (PPL)"
    )
    bars2 = ax2.bar(
        x + width / 2, loss_ft, width, color="#C44E52", label="Cross-Entropy Loss"
    )

    # Configure axes
    ax1.set_ylabel("Perplexity (PPL)", color="#4C72B0", fontweight="bold")
    ax2.set_ylabel("Cross-Entropy Loss", color="#C44E52", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)

    ax1.set_ylim(0, 5)
    ax2.set_ylim(0, 2)

    # Value labels
    for bar in bars1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            color="#4C72B0",
        )
    for bar in bars2:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            color="#C44E52",
        )

    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    save_path: str = os.path.join(output_dir, "model_capacity_dual.png")
    plt.savefig(save_path)
    plt.close(fig)
    logger.info("Saved to %s", save_path)


def plot_data_scaling(output_dir: str) -> None:
    """
    Generate a line chart illustrating data scaling laws with dual axes.
    """
    logger.info("Generating data scaling law plot (Dual Axis).")
    x_numeric: List[int] = [0, 10, 50, 100]
    y_ppl: List[float] = [9.92, 4.45, 3.86, 3.81]
    y_loss: List[float] = [2.2948, 1.4932, 1.3511, 1.3365]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        x_numeric,
        y_ppl,
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color="#4C72B0",
        label="Perplexity (PPL)",
    )
    line2 = ax2.plot(
        x_numeric,
        y_loss,
        marker="o",
        linestyle="--",
        linewidth=2,
        markersize=8,
        color="#C44E52",
        label="Cross-Entropy Loss",
    )

    ax1.set_xlabel("Training Data Fraction (%)")
    ax1.set_ylabel("Perplexity (PPL)", color="#4C72B0", fontweight="bold")
    ax2.set_ylabel("Cross-Entropy Loss", color="#C44E52", fontweight="bold")

    ax1.set_xticks([0, 10, 50, 100])
    ax1.set_xticklabels(["0%\n(Zero-Shot)", "10%", "50%", "100%"])

    ax1.set_ylim(2, 11)
    ax2.set_ylim(1.0, 2.5)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    save_path: str = os.path.join(output_dir, "data_scaling_dual.png")
    plt.savefig(save_path)
    plt.close(fig)
    logger.info("Saved to %s", save_path)


def plot_layer_ablation(output_dir: str) -> None:
    """
    Generate a grouped bar chart for the layer-wise decoupling ablation study.
    """
    logger.info("Generating layer-wise ablation plot (Dual Axis).")
    strategies: List[str] = [
        "Full Fine-Tuning\n(All Layers)",
        "Train Bottom\n(Freeze Top)",
        "Train Top\n(Freeze Bottom)",
    ]
    ppl_ablation: List[float] = [4.45, 4.64, 5.02]
    loss_ablation: List[float] = [1.4932, 1.5350, 1.6126]

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(
        x - width / 2, ppl_ablation, width, color="#8172B3", label="Perplexity (PPL)"
    )
    bars2 = ax2.bar(
        x + width / 2, loss_ablation, width, color="#64B5CD", label="Cross-Entropy Loss"
    )

    ax1.set_ylabel("Perplexity (PPL)", color="#8172B3", fontweight="bold")
    ax2.set_ylabel("Cross-Entropy Loss", color="#64B5CD", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)

    ax1.set_ylim(0, 6)
    ax2.set_ylim(0, 2)

    for bar in bars1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            color="#8172B3",
        )
    for bar in bars2:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            color="#64B5CD",
        )

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    save_path: str = os.path.join(output_dir, "layer_ablation_dual.png")
    plt.savefig(save_path)
    plt.close(fig)
    logger.info("Saved to %s", save_path)


def main() -> None:
    """
    Main execution function.
    """
    output_directory: str = "./plots"
    os.makedirs(output_directory, exist_ok=True)

    configure_academic_style()

    plot_model_capacity(output_directory)
    plot_data_scaling(output_directory)
    plot_layer_ablation(output_directory)

    logger.info("All visualization artifacts generated successfully.")


if __name__ == "__main__":
    main()
