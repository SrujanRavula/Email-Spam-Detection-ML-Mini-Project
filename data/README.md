# Dataset Documentation: Email Spam Classification Dataset

## Overview
This benchmark corpus is compiled from the genuine, public academic email archives of the **Apache SpamAssassin Public Corpus** and the **Enron Email Dataset**. It contains real-world, full-length email communications representing both authentic corporate/administrative dialogue (Ham) and unsolicited bulk communications, advance-fee frauds, phishing schemes, and commercial marketing (Spam).

## Dataset Source & Attribution
- **Primary Source:** Apache SpamAssassin Public Corpus ([SpamAssassin Official Archive](https://spamassassin.apache.org/old/publiccorpus/))
- **Secondary Source:** Enron Email Dataset (Federal Energy Regulatory Commission)
- **Kaggle / Public Mirror:** [Email Spam Classification Dataset CSV](https://www.kaggle.com/datasets/karthickveerakumar/spam-filter)
- **Direct GitHub Benchmark Mirror:** `https://raw.githubusercontent.com/amankharwal/Email-spam-detection/master/emails.csv`

## Dataset Directory Structure
```
data/
├── raw/
│   └── emails.csv          # Pristine, unmodified raw dataset (5,728 records, 8.96 MB)
└── processed/
    └── cleaned_emails.csv  # Cleansed, deduplicated, and normalized corpus (5,695 records, 15.58 MB)
```

## Schema & Attributes
The raw dataset consists of two primary columns:

| Column Name | Data Type | Description |
|---|---|---|
| `text` | `string` | Complete raw ASCII email text, including subject lines and message body. |
| `spam` | `integer` | Binary supervisory ground-truth label (`0` = Legitimate / Ham, `1` = Spam). |

## Dataset Audit & Statistics
Empirical statistics computed during the automated data audit:

| Structural Attribute | Raw Dataset | Cleaned & Deduplicated Dataset |
|---|---|---|
| **Total Records** | 5,728 | 5,695 |
| **Legitimate Emails (Ham, `0`)** | 4,360 (76.12%) | 4,327 (75.98%) |
| **Spam Emails (Spam, `1`)** | 1,368 (23.88%) | 1,368 (24.02%) |
| **Duplicate Records** | 33 (0.58%) | 0 (Fully resolved) |
| **Missing / Null Values** | 0 (0.00%) | 0 (0.00%) |
| **Disk Footprint** | 8,960,484 bytes (~8.96 MB) | 15,587,858 bytes (~15.58 MB) |

## Stratified Train-Test Partitioning (80/20)
To prevent data leakage while maintaining balanced class ratios, the cleaned corpus is partitioned using stratified sampling (`random_state=42`):

- **Training Partition (80%):** 4,556 emails
  - Legitimate (Ham): 3,462 (75.99%)
  - Spam: 1,094 (24.01%)
- **Testing Partition (20%):** 1,139 emails (strictly held out until final evaluation)
  - Legitimate (Ham): 865 (75.94%)
  - Spam: 274 (24.06%)

## Reproducibility & Acquisition
The dataset is automatically verified and acquired by the pipeline if missing:
```bash
python main.py --preprocess
```
Or programmatically via:
```python
from src.data_loader import download_dataset
download_dataset()
```
