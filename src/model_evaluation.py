"""
Model Evaluation Module
Evaluates all 10 trained Machine Learning models on the untouched test partition.
Calculates Accuracy, Spam Precision, Spam Recall, Spam F1-Score, ROC-AUC,
False Positives (FP), False Negatives (FN), Training Time, and Inference Time.
Generates individual classification reports and confusion matrices.
Follows Chapter 11 and Table 11.1 of the project technical specifications.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils import setup_logger

logger = setup_logger("ModelEvaluation")


def evaluate_single_model(
    name: str,
    pipeline: Any,
    X_test: pd.Series,
    y_test: pd.Series,
    train_time: float = 0.0,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates a single model pipeline on the test dataset.
    Computes all standard metrics and exports diagnostic artifacts.
    """
    # Measure prediction throughput
    start_time = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    pred_time = time.perf_counter() - start_time

    # Probability estimation for ROC-AUC
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            pass
    elif hasattr(pipeline, "decision_function"):
        try:
            dec_func = pipeline.decision_function(X_test)
            y_proba = (dec_func - dec_func.min()) / (dec_func.max() - dec_func.min() + 1e-10)
        except Exception:
            pass

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=config.SPAM_LABEL, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=config.SPAM_LABEL, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=config.SPAM_LABEL, zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else float("nan")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report_str = classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate (Ham)", "Spam"],
        digits=4,
    )

    clean_name = name.lower().replace(" ", "_").replace("-", "_")

    if save_artifacts:
        config.CLASSIFICATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        config.CONFUSION_MATRICES_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Save classification report
        rep_file = config.CLASSIFICATION_REPORTS_DIR / f"{clean_name}_report.txt"
        with open(rep_file, "w", encoding="utf-8") as f:
            f.write(f"Classification Report: {name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(report_str)
            f.write("\n\nConfusion Matrix:\n")
            f.write(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")
            f.write(f"Accuracy: {acc:.4f}, Spam Precision: {prec:.4f}, Spam Recall: {rec:.4f}, Spam F1: {f1:.4f}, ROC-AUC: {auc:.4f}\n")
            f.write(f"Train Time: {train_time:.3f}s, Inference Time: {pred_time:.4f}s\n")

        # 2. Save confusion matrix figure
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted Ham", "Predicted Spam"],
            yticklabels=["Actual Ham", "Actual Spam"],
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(f"Confusion Matrix: {name}\n(Acc: {acc*100:.2f}%, Spam F1: {f1:.4f})")
        cm_file = config.CONFUSION_MATRICES_DIR / f"{clean_name}_cm.png"
        plt.tight_layout()
        plt.savefig(cm_file, dpi=300, bbox_inches="tight")
        plt.close()

    metrics = {
        "model_name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "train_time": float(train_time),
        "prediction_time": float(pred_time),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    return metrics


def evaluate_all_models(
    trained_records: Dict[str, Dict[str, Any]],
    X_test: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates all trained algorithms sequentially on the untouched test partition.
    """
    eval_results: Dict[str, Dict[str, Any]] = {}
    logger.info("=" * 65)
    logger.info(f"EVALUATING ALL {len(trained_records)} MODELS ON UNTOUCHED TEST SET (N={len(X_test):,})")
    logger.info("=" * 65)

    for name, item in trained_records.items():
        pipeline = item["pipeline"]
        train_time = item["train_time"]
        res = evaluate_single_model(name, pipeline, X_test, y_test, train_time=train_time)
        eval_results[name] = res
        logger.info(
            f"{name:<26} -> Acc: {res['accuracy']:.4f} | Prec: {res['precision']:.4f} | "
            f"Rec: {res['recall']:.4f} | F1: {res['f1_score']:.4f} | AUC: {res['roc_auc']:.4f} | "
            f"FP: {res['false_positives']} | FN: {res['false_negatives']}"
        )

    logger.info("=" * 65)
    return eval_results
