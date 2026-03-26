#!/usr/bin/env python3
"""
Plot ablation Average accuracy as a vertical bar chart.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

    labels = ["Adam (Baseline)", "DSCL Only", "ADGM Only", "DASM (Full)"]
    values = [82.05, 89.13, 90.68, 93.06]

    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    colors = ["#F4A261", "#E9C46A", "#A7C957", "#8ECAE6"]
    bars = ax.bar(x_pos, values, color=colors, edgecolor="#222222", linewidth=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(80, 95.0)
    ax.margins(y=0.02)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{val:.2f}",
            va="bottom",
            ha="center",
            fontsize=10,
            color="#222222",
        )

    ax.set_title(
        "Ablation Study: Average Accuracy",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()

    out_path = Path(__file__).with_name("ablation_avg_accuracy.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
