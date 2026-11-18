# Email Spam Detection System Using Machine Learning

An end-to-end Machine Learning system for automated email spam classification, benchmarking 10 supervised classification algorithms on an authentic email corpus.

## System Features
- Full 16-stage Machine Learning Life Cycle implementation.
- Preprocessing pipeline with domain-aware tokenization (URLs, emails, currency symbols, numbers).
- Feature extraction using CountVectorizer and TfidfVectorizer (5,000 sparse features).
- Benchmark comparison across 10 supervised ML models:
  1. Support Vector Machine (LinearSVC with probability calibration)
  2. Logistic Regression
  3. Random Forest
  4. K-Nearest Neighbors
  5. Multinomial Naive Bayes
  6. Complement Naive Bayes
  7. XGBoost
  8. Gradient Boosting
  9. Decision Tree
  10. Bernoulli Naive Bayes
- 5-fold Stratified Cross-Validation hyperparameter tuning on training data.
- Complete winning model pipeline serialized with Joblib.
- Sub-millisecond real-time terminal inference CLI.
- 17 automated unit and integration tests with Pytest.

## Empirical Benchmark Results (Test Partition: N=1,139)

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Positives | False Negatives |
|---|---|---|---|---|---|---|---|
| **Support Vector Machine (Tuned)** | **99.56%** | **99.27%** | **98.91%** | **0.9909** | **0.9998** | **2** | **3** |
| Logistic Regression | 98.86% | 99.25% | 95.99% | 0.9759 | 0.9995 | 2 | 11 |
| Random Forest | 98.60% | 99.62% | 94.53% | 0.9700 | 0.9987 | 1 | 15 |
| K-Nearest Neighbors | 98.33% | 96.70% | 96.35% | 0.9653 | 0.9962 | 9 | 10 |
| Multinomial Naive Bayes | 98.24% | 97.04% | 95.62% | 0.9632 | 0.9988 | 8 | 12 |
| Complement Naive Bayes | 98.16% | 93.77% | 98.91% | 0.9627 | 0.9993 | 18 | 3 |
| XGBoost | 97.98% | 93.73% | 98.18% | 0.9590 | 0.9981 | 18 | 5 |
| Gradient Boosting | 97.98% | 96.65% | 94.89% | 0.9576 | 0.9972 | 9 | 14 |
| Decision Tree | 96.40% | 90.59% | 94.89% | 0.9269 | 0.9588 | 27 | 14 |
| Bernoulli Naive Bayes | 93.50% | 79.07% | 99.27% | 0.8803 | 0.9987 | 72 | 2 |

## Installation & Setup

1. Open PowerShell in the project directory:
   ```powershell
   cd C:\Users\Userabcd\Desktop\Email_Spam_Detection_ML
   ```

2. Activate the virtual environment:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install required packages:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage Commands

- **Launch Interactive Menu:**
  ```powershell
  python main.py
  ```

- **Run Complete 16-Stage Pipeline:**
  ```powershell
  python main.py --all
  ```

- **Run Exploratory Data Analysis:**
  ```powershell
  python main.py --eda
  ```

- **Run Data Preprocessing & Splitting:**
  ```powershell
  python main.py --preprocess
  ```

- **Train and Benchmark All 10 Models:**
  ```powershell
  python main.py --train
  ```

- **Run 5-Fold Stratified Tuning:**
  ```powershell
  python main.py --tune
  ```

- **Run Automated Pytest Suite:**
  ```powershell
  python main.py --test
  ```

- **Predict a Single Email:**
  ```powershell
  python main.py --predict "Subject: Urgent Winner Notification! You have won $500,000 cash in our lottery! Claim here: http://lottery-claim.com"



  ## Demo Video

Watch the complete working demonstration of the Email Spam Detection System.

 [▶ Watch Email Spam Detection Demo](demo/email_spam_detection_demo.mp4)
  ```
