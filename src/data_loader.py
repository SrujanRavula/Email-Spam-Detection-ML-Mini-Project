"""
Data Loader Module
Responsible for downloading the genuine benchmark dataset, loading raw records,
and performing structural data audits.
"""

import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils import setup_logger

logger = setup_logger("DataLoader")


def download_dataset(force: bool = False) -> str:
    """
    Ensures the raw dataset exists at data/raw/emails.csv.
    If missing, downloads it programmatically from public academic mirrors.
    """
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_path = config.RAW_DATA_PATH

    if target_path.exists() and not force:
        logger.info(f"Raw dataset already exists at {target_path} ({target_path.stat().st_size} bytes).")
        return str(target_path)

    logger.info(f"Downloading benchmark dataset from {config.DATASET_SOURCE_URL}...")
    try:
        urllib.request.urlretrieve(config.DATASET_SOURCE_URL, str(target_path))
        logger.info(f"Successfully downloaded dataset to {target_path}.")
    except Exception as e:
        logger.warning(f"Primary download failed ({e}). Attempting backup URL...")
        try:
            urllib.request.urlretrieve(config.DATASET_BACKUP_URL, str(target_path))
            logger.info(f"Successfully downloaded dataset from backup source to {target_path}.")
        except Exception as e2:
            raise RuntimeError(
                f"Failed to acquire dataset from remote sources: {e2}. "
                f"Please place 'emails.csv' inside '{config.RAW_DATA_DIR}' manually."
            )

    return str(target_path)


def load_raw_data() -> pd.DataFrame:
    """
    Loads raw email records from CSV.
    Validates required columns and non-emptiness.
    """
    if not config.RAW_DATA_PATH.exists():
        download_dataset()

    logger.info(f"Loading raw dataset from {config.RAW_DATA_PATH}")
    df = pd.read_csv(config.RAW_DATA_PATH)

    # Validate schema
    for col in config.DATASET_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from raw dataset.")

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    logger.info(f"Raw dataset loaded. Shape: {df.shape}")
    return df


def audit_dataset(df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Conducts a structural data audit on the raw dataset.
    Returns metadata matching Table 5.1 in the report.
    """
    if df is None:
        df = load_raw_data()

    total_records = len(df)
    duplicate_rows = int(df.duplicated().sum())
    missing_fields = int(df.isnull().sum().sum())
    ham_count = int((df["spam"] == config.HAM_LABEL).sum())
    spam_count = int((df["spam"] == config.SPAM_LABEL).sum())
    disk_size = config.RAW_DATA_PATH.stat().st_size if config.RAW_DATA_PATH.exists() else 0

    audit_summary = {
        "total_records": total_records,
        "duplicate_rows": duplicate_rows,
        "clean_unique_records": total_records - duplicate_rows,
        "missing_fields": missing_fields,
        "ham_count": ham_count,
        "ham_percentage": (ham_count / total_records) * 100 if total_records else 0,
        "spam_count": spam_count,
        "spam_percentage": (spam_count / total_records) * 100 if total_records else 0,
        "disk_bytes": disk_size,
    }

    logger.info("=" * 60)
    logger.info("DATASET STRUCTURAL METADATA & DATA QUALITY AUDIT")
    logger.info("=" * 60)
    logger.info(f"Total Raw Records       : {total_records:,}")
    logger.info(f"Legitimate Emails (Ham) : {ham_count:,} ({audit_summary['ham_percentage']:.2f}%)")
    logger.info(f"Spam Emails             : {spam_count:,} ({audit_summary['spam_percentage']:.2f}%)")
    logger.info(f"Missing / Null Values   : {missing_fields}")
    logger.info(f"Duplicate Messages      : {duplicate_rows}")
    logger.info(f"Cleaned Unique Target   : {audit_summary['clean_unique_records']:,}")
    logger.info(f"Disk Storage Footprint  : {disk_size:,} bytes ({disk_size / (1024*1024):.2f} MB)")
    logger.info("=" * 60)

    return audit_summary


if __name__ == "__main__":
    download_dataset()
    raw_df = load_raw_data()
    audit_dataset(raw_df)
