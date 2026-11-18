"""
Configuration Module for Email Spam Detection System
Defines all global constants, file paths, and hyperparameters.
Follows the experimental specifications of the academic mini-project report.
"""

from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
CLASSIFICATION_REPORTS_DIR = RESULTS_DIR / "classification_reports"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

# File Paths
RAW_DATA_PATH = RAW_DATA_DIR / "emails.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_emails.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_spam_detection_model.joblib"
BEST_METADATA_PATH = MODELS_DIR / "best_model_metadata.json"
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
MODEL_COMPARISON_TXT = RESULTS_DIR / "model_comparison.txt"
BEST_MODEL_SUMMARY_TXT = RESULTS_DIR / "best_model_summary.txt"
LOG_FILE_PATH = LOGS_DIR / "spam_detection.log"
FINAL_PROJECT_REPORT = REPORTS_DIR / "final_project_report.md"
IMPLEMENTATION_SUMMARY = REPORTS_DIR / "implementation_summary.md"

# Dataset Details & Remote URLs
DATASET_SOURCE_URL = (
    "https://raw.githubusercontent.com/amankharwal/Email-spam-detection/master/emails.csv"
)
DATASET_BACKUP_URL = (
    "https://raw.githubusercontent.com/datasets/email-spam/master/emails.csv"
)
DATASET_COLUMNS = ["text", "spam"]
HAM_LABEL = 0
SPAM_LABEL = 1

# Reproducibility & Splitting
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Expected Dataset Counts (Benchmark baseline)
EXPECTED_RAW_COUNT = 5728
EXPECTED_DUPLICATE_COUNT = 33
EXPECTED_CLEANED_COUNT = 5695
EXPECTED_HAM_COUNT = 4327
EXPECTED_SPAM_COUNT = 1368
EXPECTED_TRAIN_COUNT = 4556
EXPECTED_TEST_COUNT = 1139

# Feature Extraction Parameters
MAX_FEATURES = 5000
MIN_DF = 2
NGRAM_RANGE_BASELINE = (1, 2)
NGRAM_RANGE_TUNED = (1, 1)
SVM_BEST_C = 5.0
SUBLINEAR_TF = True
NORM = "l2"

# 10 Machine Learning Algorithm Configurations (Baseline)
MODEL_CONFIGS = {
    "Multinomial Naive Bayes": {
        "alpha": 1.0,
        "fit_prior": True,
        "feature_type": "tfidf",
    },
    "Bernoulli Naive Bayes": {
        "alpha": 1.0,
        "binarize": 0.0,
        "feature_type": "count",
    },
    "Complement Naive Bayes": {
        "alpha": 1.0,
        "norm": True,
        "feature_type": "tfidf",
    },
    "Logistic Regression": {
        "C": 1.0,
        "max_iter": 1000,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
        "feature_type": "tfidf",
    },
    "Support Vector Machine": {
        "C": 1.0,
        "cv_calibration": 3,
        "random_state": RANDOM_STATE,
        "feature_type": "tfidf",
    },
    "Decision Tree": {
        "max_depth": 35,
        "min_samples_split": 5,
        "random_state": RANDOM_STATE,
        "feature_type": "tfidf",
    },
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": 35,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "feature_type": "tfidf",
    },
    "K-Nearest Neighbors": {
        "n_neighbors": 5,
        "metric": "cosine",
        "algorithm": "brute",
        "feature_type": "tfidf",
    },
    "Gradient Boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4,
        "random_state": RANDOM_STATE,
        "feature_type": "tfidf",
    },
    "XGBoost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "feature_type": "tfidf",
    },
}

# Hyperparameter Tuning Search Grids
TUNING_GRIDS = {
    "Support Vector Machine": {
        "tfidf__max_features": [4000, 5000, 6000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__estimator__C": [0.1, 1.0, 5.0],
    },
    "Logistic Regression": {
        "tfidf__max_features": [4000, 5000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 5.0, 10.0],
    },
    "Multinomial Naive Bayes": {
        "tfidf__max_features": [4000, 5000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__alpha": [0.1, 0.5, 1.0, 2.0],
    },
}

CV_FOLDS = 5
CV_SCORING = "f1"
