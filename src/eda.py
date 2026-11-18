"""
Exploratory Data Analysis (EDA) Module
Generates statistical analyses and 4 publication-quality visualization figures
matching Chapter 5 (Figures 5.1 to 5.4) of the project report.
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils import setup_logger

logger = setup_logger("EDA")

# Styling configurations
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
    }
)


def compute_text_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes text length (characters) and word count for raw and cleaned text.
    """
    df = df.copy()
    if "char_length" not in df.columns:
        df["char_length"] = df["text"].astype(str).apply(len)
    if "word_count" not in df.columns:
        df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))
    return df


def generate_graphs(df: pd.DataFrame = None) -> Tuple[str, str, str, str]:
    """
    Generates and saves the 4 primary EDA diagnostic figures in results/graphs/:
    - Figure 5.1: Class Breakdown (Frequency Count + Proportion Pie)
    - Figure 5.2: Character Length and Word Count Density Distributions
    - Figure 5.3: Comparative Boxplots
    - Figure 5.4: Top 15 Distinctive Vocabulary Terms (Ham vs. Spam)
    """
    from src.data_preprocessing import load_processed_data

    if df is None:
        df = load_processed_data()

    df = compute_text_statistics(df)
    config.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    ham_df = df[df["spam"] == config.HAM_LABEL]
    spam_df = df[df["spam"] == config.SPAM_LABEL]

    # -------------------------------------------------------------
    # Figure 5.1: Email Spam vs. Ham Class Distribution Breakdown
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    counts = [len(ham_df), len(spam_df)]
    labels = ["Legitimate (Ham)", "Spam"]
    colors = ["#2b7bba", "#d9534f"]

    # Bar chart
    bars = ax1.bar(labels, counts, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 50,
            f"{int(height):,}\n({height/len(df)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax1.set_title("Class Frequency Count", pad=12)
    ax1.set_ylabel("Number of Emails")
    ax1.set_ylim(0, max(counts) * 1.18)

    # Donut / Pie chart
    ax2.pie(
        counts,
        labels=labels,
        autopct="%1.2f%%",
        startangle=40,
        colors=colors,
        explode=(0, 0.08),
        wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2),
    )
    ax2.set_title("Class Proportion Breakdown", pad=12)

    fig.suptitle("Figure 5.1: Email Spam vs. Ham Class Distribution Breakdown", y=1.02)
    fig_5_1_path = config.GRAPHS_DIR / "figure_5_1_class_distribution.png"
    plt.tight_layout()
    plt.savefig(fig_5_1_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 5.1 to {fig_5_1_path}")

    # -------------------------------------------------------------
    # Figure 5.2: Character Length and Word Count Density Distributions
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Cap outliers at 95th percentile for readable visualization
    char_p95 = int(df["char_length"].quantile(0.95))
    word_p95 = int(df["word_count"].quantile(0.95))

    sns.histplot(
        data=df[df["char_length"] <= char_p95],
        x="char_length",
        hue="spam",
        bins=50,
        kde=True,
        ax=ax1,
        palette={config.HAM_LABEL: "#2b7bba", config.SPAM_LABEL: "#d9534f"},
        alpha=0.4,
    )
    ax1.set_title(f"Email Character Length Distribution (<=95th percentile: {char_p95})")
    ax1.set_xlabel("Character Count")
    ax1.set_ylabel("Email Count")
    ax1.legend(["Spam (1)", "Ham (0)"])

    sns.histplot(
        data=df[df["word_count"] <= word_p95],
        x="word_count",
        hue="spam",
        bins=50,
        kde=True,
        ax=ax2,
        palette={config.HAM_LABEL: "#2b7bba", config.SPAM_LABEL: "#d9534f"},
        alpha=0.4,
    )
    ax2.set_title(f"Email Word Count Distribution (<=95th percentile: {word_p95})")
    ax2.set_xlabel("Word Count")
    ax2.set_ylabel("Email Count")
    ax2.legend(["Spam (1)", "Ham (0)"])

    fig.suptitle("Figure 5.2: Email Character Length and Word Count Density Distributions", y=1.02)
    fig_5_2_path = config.GRAPHS_DIR / "figure_5_2_length_distributions.png"
    plt.tight_layout()
    plt.savefig(fig_5_2_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 5.2 to {fig_5_2_path}")

    # -------------------------------------------------------------
    # Figure 5.3: Spam vs. Ham Message Length Comparative Boxplots
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    palette_dict = {
        config.HAM_LABEL: "#2b7bba",
        config.SPAM_LABEL: "#d9534f",
        str(config.HAM_LABEL): "#2b7bba",
        str(config.SPAM_LABEL): "#d9534f",
    }

    sns.boxplot(
        x="spam",
        y="char_length",
        hue="spam",
        data=df[df["char_length"] <= char_p95],
        ax=ax1,
        palette=palette_dict,
        legend=False,
    )
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Legitimate (Ham)", "Spam"])
    ax1.set_title("Character Count Comparison")
    ax1.set_xlabel("Email Category")
    ax1.set_ylabel("Character Length")

    sns.boxplot(
        x="spam",
        y="word_count",
        hue="spam",
        data=df[df["word_count"] <= word_p95],
        ax=ax2,
        palette=palette_dict,
        legend=False,
    )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Legitimate (Ham)", "Spam"])
    ax2.set_title("Word Count Comparison")
    ax2.set_xlabel("Email Category")
    ax2.set_ylabel("Word Count")

    fig.suptitle("Figure 5.3: Spam vs. Ham Message Length Comparative Boxplots", y=1.02)
    fig_5_3_path = config.GRAPHS_DIR / "figure_5_3_length_boxplots.png"
    plt.tight_layout()
    plt.savefig(fig_5_3_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 5.3 to {fig_5_3_path}")

    # -------------------------------------------------------------
    # Figure 5.4: Top 15 Distinctive Vocabulary Terms (Ham vs. Spam)
    # -------------------------------------------------------------
    def get_top_tokens(series: pd.Series, top_k: int = 15):
        counter = Counter()
        for text in series.dropna():
            counter.update(str(text).split())
        return counter.most_common(top_k)

    cleaned_ham_col = df[df["spam"] == config.HAM_LABEL]["cleaned_text"]
    cleaned_spam_col = df[df["spam"] == config.SPAM_LABEL]["cleaned_text"]

    top_ham = get_top_tokens(cleaned_ham_col, 15)
    top_spam = get_top_tokens(cleaned_spam_col, 15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ham_words, ham_freqs = zip(*reversed(top_ham))
    ax1.barh(ham_words, ham_freqs, color="#2b7bba", edgecolor="black", alpha=0.85)
    ax1.set_title("Top 15 Distinctive Words in Legitimate (Ham) Emails")
    ax1.set_xlabel("Occurrence Frequency")

    spam_words, spam_freqs = zip(*reversed(top_spam))
    ax2.barh(spam_words, spam_freqs, color="#d9534f", edgecolor="black", alpha=0.85)
    ax2.set_title("Top 15 Distinctive Words in Spam Emails")
    ax2.set_xlabel("Occurrence Frequency")

    fig.suptitle(
        "Figure 5.4: Top 15 Distinctive Vocabulary Terms in Legitimate vs. Spam Emails",
        y=1.02,
    )
    fig_5_4_path = config.GRAPHS_DIR / "figure_5_4_top_vocabulary.png"
    plt.tight_layout()
    plt.savefig(fig_5_4_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 5.4 to {fig_5_4_path}")

    return str(fig_5_1_path), str(fig_5_2_path), str(fig_5_3_path), str(fig_5_4_path)


def run_eda() -> Dict[str, Any]:
    """Runs complete EDA routine, logs metrics, and generates visual artifacts."""
    from src.data_preprocessing import load_processed_data

    df = load_processed_data()
    df = compute_text_statistics(df)

    ham_df = df[df["spam"] == config.HAM_LABEL]
    spam_df = df[df["spam"] == config.SPAM_LABEL]

    eda_summary = {
        "total_emails": len(df),
        "ham_emails": len(ham_df),
        "spam_emails": len(spam_df),
        "ham_mean_chars": float(ham_df["char_length"].mean()),
        "ham_median_chars": float(ham_df["char_length"].median()),
        "spam_mean_chars": float(spam_df["char_length"].mean()),
        "spam_median_chars": float(spam_df["char_length"].median()),
        "ham_mean_words": float(ham_df["word_count"].mean()),
        "spam_mean_words": float(spam_df["word_count"].mean()),
    }

    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS (EDA) SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Unique Emails Analyzed: {eda_summary['total_emails']:,}")
    logger.info(f"Ham Emails: {eda_summary['ham_emails']:,} | Spam Emails: {eda_summary['spam_emails']:,}")
    logger.info(f"Ham Avg Characters: {eda_summary['ham_mean_chars']:.1f} (Median: {eda_summary['ham_median_chars']:.1f})")
    logger.info(f"Spam Avg Characters: {eda_summary['spam_mean_chars']:.1f} (Median: {eda_summary['spam_median_chars']:.1f})")
    logger.info(f"Ham Avg Words: {eda_summary['ham_mean_words']:.1f} | Spam Avg Words: {eda_summary['spam_mean_words']:.1f}")
    logger.info("Generating publication-quality graphs...")
    generate_graphs(df)
    logger.info("EDA completed successfully.")
    logger.info("=" * 60)

    return eda_summary


if __name__ == "__main__":
    run_eda()
