"""
Test Suite for Data Pipeline
Validates dataset loading, structural schema, deduplication, stratified splitting,
and feature matrix extraction.
Follows Chapter 18 and Table 18.1 of the project technical specifications.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_loader import load_raw_data, audit_dataset
from src.data_preprocessing import load_processed_data, split_train_test
from src.feature_extraction import fit_transform_train_test


def test_raw_dataset_exists_and_loads():
    """Verifies that the raw emails.csv dataset exists and loads with correct dimensions (5,728, 2)."""
    assert config.RAW_DATA_PATH.exists(), f"Raw dataset missing at {config.RAW_DATA_PATH}"
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape == (5728, 2)


def test_dataset_columns_and_labels():
    """Verifies that the dataset contains required columns and binary target values {0, 1}."""
    df = load_raw_data()
    assert "text" in df.columns
    assert "spam" in df.columns
    unique_labels = set(df["spam"].unique())
    assert unique_labels == {0, 1}


def test_duplicate_removal():
    """Verifies that exactly 33 duplicate rows are removed, yielding 5,695 unique records."""
    df_raw = load_raw_data()
    assert int(df_raw.duplicated().sum()) == 33
    df_clean = load_processed_data()
    assert len(df_clean) == 5695
    assert int(df_clean.duplicated().sum()) == 0


def test_split_proportions_and_no_leakage():
    """Verifies exact 80/20 train-test split (4,556 train vs 1,139 test) and stratification."""
    X_train, X_test, y_train, y_test = split_train_test()
    assert len(X_train) == 4556
    assert len(X_test) == 1139
    assert len(y_train) == 4556
    assert len(y_test) == 1139

    # Verify class balance
    train_spam_ratio = (y_train == 1).mean()
    test_spam_ratio = (y_test == 1).mean()
    assert abs(train_spam_ratio - test_spam_ratio) < 0.005


def test_feature_vectorizer_dimensions_and_sparsity():
    """Verifies that TF-IDF produces valid sparse matrices of (4556, 5000) with >97% sparsity."""
    X_train, X_test, _, _ = split_train_test()
    X_train_vec, X_test_vec, vec, meta = fit_transform_train_test(
        X_train, X_test, method="tfidf"
    )
    assert X_train_vec.shape == (4556, 5000)
    assert X_test_vec.shape == (1139, 5000)
    assert len(vec.vocabulary_) == 5000
    assert meta["sparsity_percent"] > 97.0
