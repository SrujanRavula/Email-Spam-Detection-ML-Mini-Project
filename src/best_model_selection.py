"""
Best Model Selection and Serialization Module
Selects the winning model based on empirical cross-validation and test benchmarks,
evaluates its final performance on the untouched test partition,
generates Figure 14.1 (Confusion Matrix of Best Model),
and serializes the complete end-to-end inference pipeline using Joblib.
Follows Chapter 14 and Table 14.1 of the project technical specifications.
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils import setup_logger
from src.feature_extraction import create_tfidf_vectorizer

logger = setup_logger("BestModelSelection")


def build_final_best_pipeline(
    C: float = config.SVM_BEST_C,
    max_features: int = config.MAX_FEATURES,
    ngram_range: Tuple[int, int] = config.NGRAM_RANGE_TUNED,
    sublinear_tf: bool = config.SUBLINEAR_TF,
) -> Pipeline:
    """
    Constructs the winning production pipeline:
    TfidfVectorizer(max_features=5000, ngram_range=(1,1), sublinear_tf=True) +
    CalibratedClassifierCV(LinearSVC(C=5.0), cv=3).
    """
    vectorizer = create_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
    )
    base_svc = LinearSVC(
        C=C,
        random_state=config.RANDOM_STATE,
        dual="auto",
    )
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)

    return Pipeline(
        [
            ("vectorizer", vectorizer),
            ("classifier", calibrated_svc),
        ]
    )


def select_and_train_best_model(
    X_train: pd.Series,
    y_train: pd.Series,
    best_estimator: Pipeline = None,
) -> Tuple[Pipeline, float]:
    """
    Fits the winning model pipeline exclusively on training data.
    """
    logger.info("=" * 65)
    logger.info("TRAINING SELECTED BEST MODEL PIPELINE (TUNED LINEARSVC + CALIBRATION)")
    logger.info("=" * 65)

    if best_estimator is None:
        best_estimator = build_final_best_pipeline()

    start_time = time.perf_counter()
    best_estimator.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    logger.info(f"Winning model trained successfully in {train_time:.3f} seconds.")
    return best_estimator, train_time


def evaluate_and_export_best_model(
    model: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    train_time: float,
) -> Dict[str, Any]:
    """
    Evaluates winning model on test data, generates Figure 14.1 confusion matrix,
    serializes model via joblib, exports metadata JSON, and writes summary TXT.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CONFUSION_MATRICES_DIR.mkdir(parents=True, exist_ok=True)
    config.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # Predictions & Latency
    start_infer = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_time = time.perf_counter() - start_infer

    y_proba = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=config.SPAM_LABEL))
    rec = float(recall_score(y_test, y_pred, pos_label=config.SPAM_LABEL))
    f1 = float(f1_score(y_test, y_pred, pos_label=config.SPAM_LABEL))
    auc = float(roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # -------------------------------------------------------------
    # Figure 14.1: Confusion Matrix Heatmap of Best Model
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Ham", "Predicted Spam"],
        yticklabels=["Actual Ham", "Actual Spam"],
        ax=ax,
        annot_kws={"size": 15, "weight": "bold"},
    )
    ax.set_title(
        "Figure 14.1: Confusion Matrix Heatmap of the Selected Best Model\n(Support Vector Machine - LinearSVC)",
        pad=14,
    )
    fig_14_1_path = config.CONFUSION_MATRICES_DIR / "figure_14_1_best_svm_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(fig_14_1_path, dpi=300, bbox_inches="tight")
    # Also save to results/graphs/ for completeness
    plt.savefig(config.GRAPHS_DIR / "figure_14_1_best_svm_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 14.1 confusion matrix to {fig_14_1_path}")

    # Serialize complete Pipeline via Joblib
    joblib.dump(model, config.BEST_MODEL_PATH)
    logger.info(f"Serialized complete winning pipeline to {config.BEST_MODEL_PATH}")

    # Compile Metadata
    metadata = {
        "model_name": "Support Vector Machine (LinearSVC with CalibratedClassifierCV)",
        "selected_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(y_test) * 4,  # ~4,556
        "test_samples": len(y_test),         # 1,139
        "hyperparameters": {
            "C": config.SVM_BEST_C,
            "max_features": config.MAX_FEATURES,
            "ngram_range": [1, 1],
            "sublinear_tf": True,
            "norm": "l2",
            "calibration": "sigmoid (Platt scaling)",
            "cv_calibration": 3,
            "random_state": config.RANDOM_STATE,
        },
        "performance_metrics": {
            "overall_accuracy": round(acc, 4),
            "spam_precision": round(prec, 4),
            "spam_recall": round(rec, 4),
            "spam_f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positive_rate": round(fp / (tn + fp), 4),
            "false_negative_rate": round(fn / (tp + fn), 4),
            "training_time_seconds": round(train_time, 3),
            "inference_time_seconds": round(infer_time, 4),
            "latency_per_email_ms": round((infer_time / len(X_test)) * 1000.0, 4),
        },
        "artifacts": {
            "serialized_model": str(config.BEST_MODEL_PATH),
            "confusion_matrix_figure": str(fig_14_1_path),
        },
    }

    with open(config.BEST_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Exported model metadata to {config.BEST_METADATA_PATH}")

    # Write summary TXT
    summary_text = (
        "=" * 70 + "\n"
        + "BEST MODEL SELECTION SUMMARY REPORT\n"
        + "=" * 70 + "\n"
        + f"Selected Algorithm       : {metadata['model_name']}\n"
        + f"Overall Accuracy         : {acc*100:.2f}%\n"
        + f"Spam Precision           : {prec*100:.2f}%\n"
        + f"Spam Recall              : {rec*100:.2f}%\n"
        + f"Spam F1-Score            : {f1:.4f}\n"
        + f"ROC-AUC Discrimination   : {auc:.4f}\n"
        + f"False Positives (FP)     : {fp} (FPR: {metadata['performance_metrics']['false_positive_rate']*100:.2f}%)\n"
        + f"False Negatives (FN)     : {fn} (FNR: {metadata['performance_metrics']['false_negative_rate']*100:.2f}%)\n"
        + f"True Positives (TP)      : {tp}\n"
        + f"True Negatives (TN)      : {tn}\n"
        + f"Total Test Predictions   : {len(X_test):,}\n"
        + f"Training Duration        : {train_time:.3f} seconds\n"
        + f"Inference Latency        : {metadata['performance_metrics']['latency_per_email_ms']:.3f} ms / email\n"
        + "=" * 70 + "\n"
    )

    with open(config.BEST_MODEL_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info(f"Exported best model summary to {config.BEST_MODEL_SUMMARY_TXT}")

    logger.info("\n" + summary_text)
    return metadata


def run_best_model_selection(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    best_estimator: Pipeline = None,
) -> Dict[str, Any]:
    """Orchestrates training, evaluation, figure generation, and serialization."""
    model, train_time = select_and_train_best_model(X_train, y_train, best_estimator)
    metadata = evaluate_and_export_best_model(model, X_test, y_test, train_time)
    return metadata


if __name__ == "__main__":
    from src.data_preprocessing import split_train_test

    X_tr, X_te, y_tr, y_te = split_train_test()
    run_best_model_selection(X_tr, y_tr, X_te, y_te)
