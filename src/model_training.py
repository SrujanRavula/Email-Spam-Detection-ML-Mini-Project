"""
Model Training Module
Implements and trains all 10 Machine Learning algorithms specified in Chapter 9,
Chapter 10, and Table 10.1 of the project technical specifications.
Each algorithm is instantiated as an end-to-end Scikit-learn Pipeline
encapsulating its feature extractor and classifier.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils import setup_logger
from src.feature_extraction import create_count_vectorizer, create_tfidf_vectorizer

logger = setup_logger("ModelTraining")


def build_model_pipelines() -> Dict[str, Pipeline]:
    """
    Constructs Scikit-learn Pipelines for all 10 algorithms with their
    designated feature representations and baseline hyperparameters.
    """
    pipelines: Dict[str, Pipeline] = {}

    # 1. Multinomial Naive Bayes
    pipelines["Multinomial Naive Bayes"] = Pipeline(
        [
            ("vectorizer", create_tfidf_vectorizer()),
            ("classifier", MultinomialNB(alpha=1.0, fit_prior=True)),
        ]
    )

    # 2. Bernoulli Naive Bayes (uses binary CountVectorizer)
    pipelines["Bernoulli Naive Bayes"] = Pipeline(
        [
            ("vectorizer", create_count_vectorizer()),
            ("classifier", BernoulliNB(alpha=1.0, binarize=0.0)),
        ]
    )

    # 3. Complement Naive Bayes (handles class imbalance)
    pipelines["Complement Naive Bayes"] = Pipeline(
        [
            ("vectorizer", create_tfidf_vectorizer()),
            ("classifier", ComplementNB(alpha=1.0, norm=True)),
        ]
    )

    # 4. Logistic Regression
    pipelines["Logistic Regression"] = Pipeline(
        [
            ("vectorizer", create_tfidf_vectorizer()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    penalty="l2",
                    solver="lbfgs",
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )

    # 5. Support Vector Machine (LinearSVC with Platt 3-fold calibration)
    base_svc = LinearSVC(
        C=1.0,
        random_state=config.RANDOM_STATE,
        dual="auto",
    )
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)
    pipelines["Support Vector Machine"] = Pipeline(
        [
            ("vectorizer", create_tfidf_vectorizer()),
            ("classifier", calibrated_svc),
        ]
    )

    # 6. Decision Tree Classifier
    pipelines["Decision Tree"] = Pipeline(
        [
            ("vectorizer", create_tfidf_vectorizer()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=35,
                    min_samples_split=5,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )

    # 7. Random Forest Classifier
    pipelines["Random Forest"] = Pipeline(
        [
            ("vectorizer", create_tfidf_vectorizer()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=35,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 8. K-Nearest Neighbors Classifier
    pipelines["K-Nearest Neighbors"] = Pipeline(
        [
            ("vectorizer", create_tfidf_vectorizer()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=5,
                    metric="cosine",
                    algorithm="brute",
                ),
            ),
        ]
    )

    # 9. Gradient Boosting Classifier
    pipelines["Gradient Boosting"] = Pipeline(
        [
            ("vectorizer", create_tfidf_vectorizer()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )

    # 10. XGBoost Classifier
    pipelines["XGBoost"] = Pipeline(
        [
            ("vectorizer", create_tfidf_vectorizer()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    eval_metric="logloss",
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return pipelines


def train_all_models(
    X_train: pd.Series, y_train: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """
    Fits all 10 machine learning pipelines on the training partition exclusively.
    Measures training durations and returns fitted pipelines.
    """
    pipelines = build_model_pipelines()
    trained_records: Dict[str, Dict[str, Any]] = {}

    total_models = len(pipelines)
    logger.info("=" * 65)
    logger.info(f"TRAINING ALL {total_models} MACHINE LEARNING ALGORITHMS")
    logger.info("=" * 65)

    idx = 1
    for name, pipeline in pipelines.items():
        logger.info(f"[{idx:02d}/{total_models}] Training {name:<26} ...")
        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - start_time
        logger.info(f"[{idx:02d}/{total_models}] Training {name:<26} ... Done ({train_time:.2f}s)")

        trained_records[name] = {
            "pipeline": pipeline,
            "train_time": train_time,
        }
        idx += 1

    logger.info("=" * 65)
    logger.info(f"All {total_models} models trained successfully without errors.")
    logger.info("=" * 65)

    return trained_records


if __name__ == "__main__":
    from src.data_preprocessing import split_train_test

    X_tr, X_te, y_tr, y_te = split_train_test()
    records = train_all_models(X_tr, y_tr)
