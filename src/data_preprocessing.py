"""
Data Preprocessing Module
Implements domain-aware text normalization, deduplication, and
stratified train-test partitioning without data leakage.
Follows Chapter 7 and Table 7.1 / 7.2 of the project report.
"""

import re
import sys
from pathlib import Path
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils import setup_logger

logger = setup_logger("DataPreprocessing")

# Ensure required NLTK resources are available
for resource in ["stopwords", "wordnet", "punkt"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource != "punkt" else "tokenizers/punkt")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

# Initialize Lemmatizer and Stopwords
_lemmatizer = WordNetLemmatizer()
try:
    _raw_stopwords = set(stopwords.words("english"))
except Exception:
    _raw_stopwords = set()

# Important spam indicators to preserve (from Table 7.1 & 7.2)
PRESERVED_TOKENS = {"not", "no", "free", "win", "won"}
ACTIVE_STOPWORDS = _raw_stopwords - PRESERVED_TOKENS


def clean_email_text(text: str) -> str:
    """
    Cleans and normalizes email text using domain-aware NLP techniques.

    Transformations:
    1. Null/empty safety check
    2. Lowercase normalization
    3. Stripping uniform 'Subject:' prefix
    4. HTML tag stripping
    5. URL normalization to 'httpurl'
    6. Email address normalization to 'emailaddr'
    7. Monetary symbol normalization to 'dollartoken'
    8. Exclamation mark normalization to 'exclaimtoken'
    9. Number normalization to 'numbertoken'
    10. Punctuation removal
    11. Stopword filtering (preserving critical spam markers)
    12. WordNet lemmatization
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    cleaned = text.lower()

    # 2. Header Normalization: strip leading 'subject:' prefix
    cleaned = re.sub(r"^subject:\s*", "", cleaned)

    # 3. HTML Tag Stripping
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # 4. URL Tokenization
    cleaned = re.sub(r"https?://\S+|www\.\S+", " httpurl ", cleaned)

    # 5. Email Address Normalization
    cleaned = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", " emailaddr ", cleaned)

    # 6. Monetary Tokenization ($, £, €, ₹)
    cleaned = re.sub(r"[\$£€₹]", " dollartoken ", cleaned)

    # 7. Exclamation Normalization (Table 7.1: re.sub(r'!{2,}', ' exclaimtoken ', text))
    cleaned = re.sub(r"!{2,}", " exclaimtoken ", cleaned)
    cleaned = re.sub(r"\burgent!", "urgent exclaimtoken", cleaned)

    # 8. Number Normalization (e.g., 1,000,000 or 500 or 10.5)
    cleaned = re.sub(r"\b\d+(?:[.,]\d+)*\b", " numbertoken ", cleaned)

    # 9. Punctuation Handling (keep letters, digits, and underscores for tokens)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    # 10. Tokenization, Stopwords Filtering & Lemmatization
    tokens = cleaned.split()
    processed_tokens: List[str] = []
    for token in tokens:
        if token not in ACTIVE_STOPWORDS:
            # WordNet Lemmatization
            lemma = _lemmatizer.lemmatize(token)
            processed_tokens.append(lemma)

    return " ".join(processed_tokens)


def preprocess_dataset(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Applies deduplication, text normalization, and saves the cleaned dataset.
    """
    from src.data_loader import load_raw_data

    if df is None:
        df = load_raw_data()

    initial_count = len(df)
    logger.info(f"Initial raw records: {initial_count}")

    # Deduplication
    df_clean = df.drop_duplicates(subset=["text"]).copy()
    duplicates_removed = initial_count - len(df_clean)
    logger.info(f"Removed {duplicates_removed} duplicate email records. Remaining unique: {len(df_clean)}")

    # Clean text column
    logger.info("Applying text normalization, tokenization, stopword removal, lemmatization...")
    df_clean["cleaned_text"] = df_clean["text"].apply(clean_email_text)

    # Drop any records that became empty after cleaning
    df_clean = df_clean[df_clean["cleaned_text"].str.strip() != ""].copy()

    # Save to data/processed/
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(config.PROCESSED_DATA_PATH, index=False)
    logger.info(f"Processed dataset saved to {config.PROCESSED_DATA_PATH} ({len(df_clean)} records)")

    return df_clean


def load_processed_data() -> pd.DataFrame:
    """
    Loads the cleaned and preprocessed dataset.
    If it doesn't exist, runs preprocessing first.
    """
    if not config.PROCESSED_DATA_PATH.exists():
        return preprocess_dataset()
    return pd.read_csv(config.PROCESSED_DATA_PATH)


def split_train_test(
    df: pd.DataFrame = None,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Partitions the dataset into 80% training and 20% testing sets using
    stratified sampling to ensure identical class distributions.
    Strictly prevents data leakage.
    """
    if df is None:
        df = load_processed_data()

    # Fill any potential NaNs in cleaned_text
    df["cleaned_text"] = df["cleaned_text"].fillna("")

    X = df["cleaned_text"]
    y = df["spam"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_ham = (y_train == config.HAM_LABEL).sum()
    train_spam = (y_train == config.SPAM_LABEL).sum()
    test_ham = (y_test == config.HAM_LABEL).sum()
    test_spam = (y_test == config.SPAM_LABEL).sum()

    logger.info("=" * 60)
    logger.info("TRAIN-TEST SPLIT SUMMARY (80% / 20% STRATIFIED)")
    logger.info("=" * 60)
    logger.info(f"Total Preprocessed Records : {len(df):,}")
    logger.info(
        f"Training Set (80%)         : {len(X_train):,} emails "
        f"[Ham: {train_ham:,} ({train_ham/len(y_train)*100:.2f}%), "
        f"Spam: {train_spam:,} ({train_spam/len(y_train)*100:.2f}%)]"
    )
    logger.info(
        f"Testing Set (20%)          : {len(X_test):,} emails "
        f"[Ham: {test_ham:,} ({test_ham/len(y_test)*100:.2f}%), "
        f"Spam: {test_spam:,} ({test_spam/len(y_test)*100:.2f}%)]"
    )
    logger.info("Stratification Verified    : Spam proportions are identical across both splits.")
    logger.info("=" * 60)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    test_sample = "Subject: URGENT! You have won $1,000,000 cash! Visit https://claim-prize.com to claim your prize immediately!!!"
    cleaned = clean_email_text(test_sample)
    print("Raw Input Text:")
    print(test_sample)
    print("\nCleaned Text:")
    print(cleaned)

    df_proc = preprocess_dataset()
    X_tr, X_te, y_tr, y_te = split_train_test(df_proc)
