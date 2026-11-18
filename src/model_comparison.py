"""
Model Comparison Module
Compiles master empirical benchmark results across all 10 algorithms,
exports CSV and formatted TXT tables, and generates comparative visualization charts
matching Chapter 12 (Figures 12.1, 12.2, 12.3) of the project technical specifications.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils import setup_logger

logger = setup_logger("ModelComparison")


def compile_comparison_dataframe(eval_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Transforms dictionary of evaluation results into a structured DataFrame,
    sorted in descending order by Spam F1-Score (matching Table 12.1).
    """
    records = []
    for name, res in eval_results.items():
        records.append(
            {
                "Algorithm": name,
                "Accuracy": round(res["accuracy"], 4),
                "Precision": round(res["precision"], 4),
                "Recall": round(res["recall"], 4),
                "F1-Score": round(res["f1_score"], 4),
                "ROC-AUC": round(res["roc_auc"], 4) if not np.isnan(res["roc_auc"]) else 0.0,
                "False Positives": int(res["false_positives"]),
                "False Negatives": int(res["false_negatives"]),
                "Train Time (s)": round(res["train_time"], 3),
                "Prediction Time (s)": round(res["prediction_time"], 4),
            }
        )

    df_comp = pd.DataFrame(records)
    df_comp = df_comp.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    return df_comp


def export_comparison_tables(df_comp: pd.DataFrame) -> Tuple[Path, Path]:
    """
    Exports the benchmark DataFrame to CSV and formatted TXT tables.
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Also support results/model_comparison/ directory if referenced
    comp_sub_dir = config.RESULTS_DIR / "model_comparison"
    comp_sub_dir.mkdir(parents=True, exist_ok=True)

    csv_path = config.MODEL_COMPARISON_CSV
    txt_path = config.MODEL_COMPARISON_TXT

    df_comp.to_csv(csv_path, index=False)
    df_comp.to_csv(comp_sub_dir / "model_comparison.csv", index=False)

    # Format ASCII table
    formatted_table = df_comp.to_string(index=False)
    header = (
        "=" * 100 + "\n"
        + "MASTER EMPIRICAL EVALUATION BENCHMARK ACROSS ALL 10 ML ALGORITHMS (TEST SET)\n"
        + "=" * 100 + "\n\n"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(header + formatted_table + "\n\n" + "=" * 100 + "\n")
    with open(comp_sub_dir / "model_comparison.txt", "w", encoding="utf-8") as f:
        f.write(header + formatted_table + "\n\n" + "=" * 100 + "\n")

    logger.info(f"Model comparison CSV saved to {csv_path}")
    logger.info(f"Model comparison TXT saved to {txt_path}")

    return csv_path, txt_path


def generate_comparison_charts(df_comp: pd.DataFrame) -> Tuple[Path, Path, Path]:
    """
    Generates comparative benchmark visualization figures:
    - Figure 12.1: Multi-Metric Performance Benchmark
    - Figure 12.2: Spam F1-Score Comparative Ranking
    - Figure 12.3: Training Latency vs. Spam F1-Score Trade-off
    """
    config.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Figure 12.1: Multi-Metric Performance Benchmark
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    algorithms = df_comp["Algorithm"].tolist()
    x = np.arange(len(algorithms))
    width = 0.18

    rects1 = ax.bar(x - 1.5 * width, df_comp["Accuracy"], width, label="Accuracy", color="#2b7bba")
    rects2 = ax.bar(x - 0.5 * width, df_comp["Precision"], width, label="Precision", color="#339966")
    rects3 = ax.bar(x + 0.5 * width, df_comp["Recall"], width, label="Recall", color="#e67e22")
    rects4 = ax.bar(x + 1.5 * width, df_comp["F1-Score"], width, label="F1-Score", color="#d9534f")

    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title("Figure 12.1: Multi-Metric Performance Benchmark Across All 10 Machine Learning Algorithms")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=30, ha="right")
    ax.set_ylim(0.75, 1.02)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig_12_1_path = config.GRAPHS_DIR / "figure_12_1_multi_metric_comparison.png"
    plt.tight_layout()
    plt.savefig(fig_12_1_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 12.1 to {fig_12_1_path}")

    # -------------------------------------------------------------
    # Figure 12.2: Spam F1-Score Comparative Ranking
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    df_f1_sorted = df_comp.sort_values(by="F1-Score", ascending=True)

    # Distinctive palette: highlight leading models with vibrant color
    colors = ["#2b7bba" if f1 < 0.98 else "#28a745" for f1 in df_f1_sorted["F1-Score"]]
    bars = ax.barh(df_f1_sorted["Algorithm"], df_f1_sorted["F1-Score"], color=colors, edgecolor="black", alpha=0.85)

    for bar in bars:
        width_val = bar.get_width()
        ax.text(
            width_val + 0.003,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width_val:.4f}",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_xlabel("Spam F1-Score")
    ax.set_xlim(0.80, 1.03)
    ax.set_title("Figure 12.2: Spam F1-Score Comparative Ranking Across All Implemented Algorithms")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    fig_12_2_path = config.GRAPHS_DIR / "figure_12_2_f1_score_ranking.png"
    plt.tight_layout()
    plt.savefig(fig_12_2_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 12.2 to {fig_12_2_path}")

    # -------------------------------------------------------------
    # Figure 12.3: Training Latency vs. Spam F1-Score Trade-off
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        df_comp["Train Time (s)"],
        df_comp["F1-Score"],
        color="#2b7bba",
        s=120,
        edgecolors="black",
        zorder=5,
    )

    for _, row in df_comp.iterrows():
        ax.annotate(
            row["Algorithm"],
            (row["Train Time (s)"], row["F1-Score"]),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=9,
            weight="semibold",
        )

    ax.set_xlabel("Training Duration (Seconds)")
    ax.set_ylabel("Spam F1-Score")
    ax.set_title("Figure 12.3: Computational Efficiency: Training Duration vs. Spam F1-Score Trade-off")
    ax.grid(True, linestyle="--", alpha=0.7)

    fig_12_3_path = config.GRAPHS_DIR / "figure_12_3_training_time_vs_f1.png"
    plt.tight_layout()
    plt.savefig(fig_12_3_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 12.3 to {fig_12_3_path}")

    return fig_12_1_path, fig_12_2_path, fig_12_3_path


def run_model_comparison(eval_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Executes the full model comparison routine and exports all artifacts."""
    df_comp = compile_comparison_dataframe(eval_results)
    export_comparison_tables(df_comp)
    generate_comparison_charts(df_comp)
    return df_comp
