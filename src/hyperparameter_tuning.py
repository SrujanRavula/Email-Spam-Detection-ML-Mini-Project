"""
Hyperparameter Tuning Module
Executes 5-fold Stratified Cross-Validation on training data exclusively
to optimize hyperparameters for top candidate models:
1. Support Vector Machine (LinearSVC with calibration)
2. Logistic Regression
3. Multinomial Naive Bayes
Follows Chapter 13 and Table 13.1 / 13.2 of the project technical specifications.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils import setup_logger
from src.feature_extraction import create_tfidf_vectorizer

logger = setup_logger("HyperparameterTuning")


def tune_candidate_model(
    name: str,
    pipeline: Pipeline,
    param_grid: Dict[str, list],
    X_train: pd.Series,
    y_train: pd.Series,
    cv_folds: int = config.CV_FOLDS,
    scoring: str = config.CV_SCORING,
) -> Tuple[GridSearchCV, Dict[str, Any]]:
    """
    Performs 5-Fold Stratified Cross-Validation Grid Search on training partition.
    Strictly preserves testing data for unbiased evaluation.
    """
    logger.info("-" * 60)
    logger.info(f"Tuning {name} using {cv_folds}-Fold Stratified CV (metric: '{scoring}')...")
    logger.info(f"Search Grid: {param_grid}")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    start_time = time.perf_counter()
    grid_search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time

    best_params = grid_search.best_params_
    best_score = float(grid_search.best_score_)

    logger.info(f"Tuning {name} completed in {elapsed:.2f}s.")
    logger.info(f"Optimal Parameters : {best_params}")
    logger.info(f"Best {cv_folds}-Fold CV F1-Score: {best_score:.4f}")

    record = {
        "candidate": name,
        "best_params": best_params,
        "best_cv_f1": best_score,
        "tuning_time_sec": elapsed,
        "best_estimator": grid_search.best_estimator_,
    }

    return grid_search, record


def run_all_tuning(
    X_train: pd.Series, y_train: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """
    Tunes all leading candidate architectures specified in Table 13.1.
    """
    tuning_results: Dict[str, Dict[str, Any]] = {}

    logger.info("=" * 65)
    logger.info("HYPERPARAMETER OPTIMIZATION (5-FOLD STRATIFIED CV ON TRAIN SET)")
    logger.info("=" * 65)

    # 1. Support Vector Machine
    base_svc = LinearSVC(random_state=config.RANDOM_STATE, dual="auto")
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)
    svm_pipe = Pipeline(
        [
            ("tfidf", create_tfidf_vectorizer()),
            ("clf", calibrated_svc),
        ]
    )
    svm_grid = {
        "tfidf__max_features": [4000, 5000, 6000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__estimator__C": [0.1, 1.0, 5.0],
    }
    _, svm_record = tune_candidate_model(
        "Support Vector Machine", svm_pipe, svm_grid, X_train, y_train
    )
    tuning_results["Support Vector Machine"] = svm_record

    # 2. Logistic Regression
    lr_pipe = Pipeline(
        [
            ("tfidf", create_tfidf_vectorizer()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    penalty="l2",
                    solver="lbfgs",
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )
    lr_grid = {
        "tfidf__max_features": [4000, 5000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 5.0, 10.0],
    }
    _, lr_record = tune_candidate_model(
        "Logistic Regression", lr_pipe, lr_grid, X_train, y_train
    )
    tuning_results["Logistic Regression"] = lr_record

    # 3. Multinomial Naive Bayes
    mnb_pipe = Pipeline(
        [
            ("tfidf", create_tfidf_vectorizer()),
            ("clf", MultinomialNB(fit_prior=True)),
        ]
    )
    mnb_grid = {
        "tfidf__max_features": [4000, 5000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__alpha": [0.1, 0.5, 1.0, 2.0],
    }
    _, mnb_record = tune_candidate_model(
        "Multinomial Naive Bayes", mnb_pipe, mnb_grid, X_train, y_train
    )
    tuning_results["Multinomial Naive Bayes"] = mnb_record

    logger.info("=" * 65)
    logger.info("Hyperparameter tuning completed successfully for all candidate models.")
    logger.info("=" * 65)

    return tuning_results


if __name__ == "__main__":
    from src.data_preprocessing import split_train_test

    X_tr, X_te, y_tr, y_te = split_train_test()
    run_all_tuning(X_tr, y_tr)
