"""
verification/regenerate_radar_chart.py
======================================
Regenerates the Multi-Metric Evaluation Radar chart (fig12_radar.png)
from results/cross_dataset_results.csv, ensuring CausalShield is correctly
labeled with 7 features: 'CausalShield (ACE) (7 feats)'.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "cross_dataset_results.csv"
OUT_PATH = RESULTS_DIR / "fig12_radar.png"


def generate_radar():
    df = pd.read_csv(CSV_PATH)

    categories = ["ROC-AUC", "PR-AUC", "Macro F1", "Accuracy", "Benign F1", "Attack F1"]
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    colors = {
        "Full-Feature XGBoost": "#c0392b",
        "MI-Selected XGBoost": "#e67e22",
        "CausalShield (ACE)": "#27ae60",
    }
    fill_alphas = {
        "Full-Feature XGBoost": 0.08,
        "MI-Selected XGBoost": 0.15,
        "CausalShield (ACE)": 0.12,
    }

    fig = plt.figure(figsize=(7, 8.5), dpi=180)
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, fontsize=12, color="#1a1a1a")
    ax.tick_params(axis="x", pad=14)

    ax.set_rlabel_position(15)
    plt.yticks(
        [0.25, 0.50, 0.75, 1.00],
        ["0.25", "0.50", "0.75", "1.00"],
        color="#7f8c8d",
        fontsize=10,
    )
    plt.ylim(0, 1.05)
    ax.grid(color="#d0d0d0", linestyle="-", linewidth=0.8, alpha=0.9)
    ax.spines["polar"].set_color("#b0b0b0")

    for _, row in df.iterrows():
        model_name = row["Model"]
        n_feats = int(row["Features"])
        values = [row[cat] for cat in categories]
        values += values[:1]

        label = f"{model_name} ({n_feats} feats)"
        c = colors.get(model_name, "#333333")
        alpha = fill_alphas.get(model_name, 0.1)

        ax.plot(angles, values, color=c, linewidth=2.5, label=label)
        ax.fill(angles, values, color=c, alpha=alpha)

    ax.set_title(
        "Multi-Metric Profile: Cross-Dataset Generalisation\n(CICIoT2023 \u2192 CICIDS2017, Zero-Shot OOD)",
        fontsize=13.5,
        fontweight="bold",
        pad=22,
        color="#0f172a",
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=True,
        facecolor="white",
        edgecolor="#b0b0b0",
        fontsize=11.5,
    )

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    docs_out = ROOT / "docs" / "fig12_radar.png"
    if (ROOT / "docs").exists():
        plt.savefig(docs_out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated: {OUT_PATH}")


if __name__ == "__main__":
    generate_radar()
