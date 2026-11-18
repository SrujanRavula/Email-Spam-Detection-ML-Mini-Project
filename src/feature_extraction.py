"""
Feature Extraction Module
Transforms cleaned text strings into high-dimensional numerical feature matrices.
Supports CountVectorizer and TfidfVectorizer with sublinear term frequency scaling.
Prevents data leakage by fitting exclusively on training data.
Follows Chapter 8 and Table 8.1 of the project technical specifications.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils import setup_logger

logger = setup_logger("FeatureExtraction")


def create_count_vectorizer(
    max_features: int = config.MAX_FEATURES,
    ngram_range: Tuple[int, int] = config.NGRAM_RANGE_BASELINE,
    min_df: int = config.MIN_DF,
) -> CountVectorizer:
    """Instantiates a CountVectorizer (Bag-of-Words) feature extractor."""
    return CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )


def create_tfidf_vectorizer(
    max_features: int = config.MAX_FEATURES,
    ngram_range: Tuple[int, int] = config.NGRAM_RANGE_BASELINE,
    min_df: int = config.MIN_DF,
    sublinear_tf: bool = config.SUBLINEAR_TF,
    norm: str = config.NORM,
) -> TfidfVectorizer:
    """
    Instantiates a TfidfVectorizer with sublinear TF scaling and L2 normalization.
    Sublinear TF: TF(t, d) = 1 + log(f_{t, d}) for f_{t, d} > 0.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        norm=norm,
    )


def analyze_feature_matrix(
    matrix: sp.csr_matrix,
    vectorizer: Any,
    name: str = "Feature Matrix",
) -> Dict[str, Any]:
    """
    Computes mathematical properties of a sparse feature matrix including
    vocabulary size, dimensions, non-zero entries, and matrix sparsity.
    """
    n_samples, n_features = matrix.shape
    total_cells = n_samples * n_features
    nnz = matrix.nnz
    sparsity = (1.0 - (nnz / total_cells)) * 100.0 if total_cells > 0 else 0.0

    analysis = {
        "name": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "nnz": nnz,
        "sparsity_percent": sparsity,
    }

    logger.info("=" * 60)
    logger.info(f"SPARSE MATRIX ANALYSIS: {name.upper()}")
    logger.info("=" * 60)
    logger.info(f"Matrix Dimensions       : ({n_samples:,}, {n_features:,})")
    logger.info(f"Vocabulary Features     : {len(vectorizer.vocabulary_):,}")
    logger.info(f"Non-Zero Entries (nnz)  : {nnz:,}")
    logger.info(f"Matrix Sparsity Ratio   : {sparsity:.2f}% (zero-valued cells)")
    logger.info(f"Sparse Storage Type     : {type(matrix).__name__}")
    logger.info("=" * 60)

    return analysis


def fit_transform_train_test(
    X_train,
    X_test,
    method: str = "tfidf",
    max_features: int = config.MAX_FEATURES,
    ngram_range: Tuple[int, int] = config.NGRAM_RANGE_BASELINE,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, Any, Dict[str, Any]]:
    """
    Fits vectorizer strictly on X_train to prevent data leakage, then
    transforms X_train and X_test.
    """
    if method == "count":
        vec = create_count_vectorizer(max_features=max_features, ngram_range=ngram_range)
    else:
        vec = create_tfidf_vectorizer(max_features=max_features, ngram_range=ngram_range)

    logger.info(f"Fitting {method.upper()} vectorizer on training data exclusively...")
    X_train_vec = vec.fit_transform(X_train)
    logger.info(f"Transforming test data using fitted {method.upper()} vectorizer...")
    X_test_vec = vec.transform(X_test)

    analysis = analyze_feature_matrix(X_train_vec, vec, f"{method.upper()} Training Matrix")

    return X_train_vec, X_test_vec, vec, analysis


if __name__ == "__main__":
    from src.data_preprocessing import split_train_test

    X_tr, X_te, y_tr, y_te = split_train_test()
    X_tr_tfidf, X_te_tfidf, tfidf_vec, tfidf_meta = fit_transform_train_test(
        X_tr, X_te, method="tfidf"
    )
    X_tr_cnt, X_te_cnt, cnt_vec, cnt_meta = fit_transform_train_test(
        X_tr, X_te, method="count"
    )
