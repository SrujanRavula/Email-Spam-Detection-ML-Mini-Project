"""
Email Spam Detection System — Central Orchestrator & CLI Interface
Author: B.Tech CSE (AI & ML) Mini-Project Implementation
Command-line Interface supporting:
  python main.py             (Interactive Terminal Menu)
  python main.py --all       (Execute Complete 16-Stage ML Life Cycle)
  python main.py --eda       (Execute Exploratory Data Analysis)
  python main.py --preprocess(Execute Text Preprocessing & Splitting)
  python main.py --train     (Train & Benchmark All 10 ML Algorithms)
  python main.py --tune      (Execute 5-Fold Stratified CV Tuning)
  python main.py --test      (Execute Automated Pytest Test Suite)
  python main.py --predict   "Your email text here"
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils import setup_logger, print_banner
from src.data_loader import download_dataset, load_raw_data, audit_dataset
from src.data_preprocessing import preprocess_dataset, split_train_test
from src.eda import run_eda
from src.feature_extraction import fit_transform_train_test
from src.model_training import train_all_models
from src.model_evaluation import evaluate_all_models
from src.model_comparison import run_model_comparison
from src.hyperparameter_tuning import run_all_tuning
from src.best_model_selection import run_best_model_selection
from src.prediction import predict_email, format_prediction_result, load_best_model

logger = setup_logger("Main")


def run_complete_ml_lifecycle():
    """Executes the complete 16-stage Machine Learning Life Cycle."""
    print_banner("EXECUTING COMPLETE 16-STAGE MACHINE LEARNING LIFE CYCLE", width=75)

    # Stages 1, 2, 3: Dataset Ingestion & Audit
    logger.info("Stage 1-3: Loading and auditing benchmark dataset...")
    download_dataset()
    raw_df = load_raw_data()
    audit_dataset(raw_df)

    # Stage 4: Exploratory Data Analysis & Visualizations
    logger.info("Stage 4: Running Exploratory Data Analysis...")
    run_eda()

    # Stages 5, 6, 7: Data Cleaning, Preprocessing & Stratified Splitting
    logger.info("Stage 5-7: Running Preprocessing & Stratified 80/20 Train-Test Split...")
    df_clean = preprocess_dataset(raw_df)
    X_train, X_test, y_train, y_test = split_train_test(df_clean)

    # Stage 8: Feature Extraction & Sparsity Analysis
    logger.info("Stage 8: Analyzing Feature Extraction and Matrix Sparsity...")
    fit_transform_train_test(X_train, X_test, method="tfidf")

    # Stages 9, 10: Model Pipelines & Training Across 10 Algorithms
    logger.info("Stage 9-10: Assembling Pipelines and Training All 10 ML Algorithms...")
    trained_records = train_all_models(X_train, y_train)

    # Stage 11: Model Evaluation on Untouched Test Set
    logger.info("Stage 11: Evaluating All 10 Models on Untouched Test Partition...")
    eval_results = evaluate_all_models(trained_records, X_test, y_test)

    # Stage 12: Comparative Benchmarking & Graphs
    logger.info("Stage 12: Compiling Comparative Benchmark Tables & Visualization Charts...")
    run_model_comparison(eval_results)

    # Stage 13: 5-Fold Stratified Cross-Validated Hyperparameter Tuning
    logger.info("Stage 13: Performing 5-Fold Stratified CV Hyperparameter Tuning on Leading Models...")
    run_all_tuning(X_train, y_train)

    # Stages 14, 15: Best Model Selection, Confusion Matrix & Joblib Serialization
    logger.info("Stage 14-15: Selecting Best Model, Generating Final Heatmap & Serializing Pipeline...")
    metadata = run_best_model_selection(X_train, y_train, X_test, y_test)

    # Stage 16: Inference Verification
    logger.info("Stage 16: Verifying Real-Time Terminal Inference...")
    sample_spam = "Subject: Urgent Winner Notification! You have won $500,000 cash in our international lottery! Claim here now: http://lottery-claim.com"
    sample_ham = "Subject: Machine Learning Seminar Schedule. Hello students, our next lecture on Support Vector Machines is scheduled for Friday at 10 AM in Room 302."

    res_spam = predict_email(sample_spam)
    print(format_prediction_result(res_spam))

    res_ham = predict_email(sample_ham)
    print(format_prediction_result(res_ham))

    print_banner("16-STAGE MACHINE LEARNING LIFE CYCLE COMPLETED SUCCESSFULLY", width=75)


def run_tests():
    """Runs the automated pytest test suite."""
    print_banner("RUNNING AUTOMATED PYTEST TEST SUITE", width=75)
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    result = subprocess.run(cmd, cwd=str(config.BASE_DIR))
    return result.returncode


def display_menu():
    """Interactive Python terminal interface."""
    while True:
        print_banner("EMAIL SPAM DETECTION SYSTEM — INTERACTIVE DASHBOARD", width=70)
        print(" [1] Predict Single Email (Spam / Ham Classification)")
        print(" [2] View Selected Best Model Metadata & Test Metrics")
        print(" [3] View 10-Algorithm Master Comparative Benchmark")
        print(" [4] Run Complete Machine Learning Life Cycle (--all)")
        print(" [5] Run Automated Unit Tests (pytest)")
        print(" [6] Exit")
        print("=" * 70)

        choice = input("Enter your option [1-6]: ").strip()

        if choice == "1":
            print("\nEnter email text to classify (Press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            email_text = "\n".join(lines).strip()
            if not email_text:
                print("\n[!] Error: Empty email input provided.\n")
                continue

            try:
                result = predict_email(email_text)
                print(format_prediction_result(result))
            except Exception as e:
                print(f"\n[!] Inference Error: {e}\n")

        elif choice == "2":
            if config.BEST_METADATA_PATH.exists():
                with open(config.BEST_METADATA_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                print("\n" + json.dumps(meta, indent=4) + "\n")
            elif config.BEST_MODEL_SUMMARY_TXT.exists():
                with open(config.BEST_MODEL_SUMMARY_TXT, "r", encoding="utf-8") as f:
                    print("\n" + f.read() + "\n")
            else:
                print("\n[!] Best model metadata not found. Please train models first via Option 4.\n")

        elif choice == "3":
            if config.MODEL_COMPARISON_TXT.exists():
                with open(config.MODEL_COMPARISON_TXT, "r", encoding="utf-8") as f:
                    print("\n" + f.read() + "\n")
            else:
                print("\n[!] Model comparison table not found. Please run model training first.\n")

        elif choice == "4":
            run_complete_ml_lifecycle()

        elif choice == "5":
            run_tests()

        elif choice == "6":
            print("\nExiting Email Spam Detection System. Goodbye!\n")
            break

        else:
            print("\n[!] Invalid selection. Please choose an option from 1 to 6.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Email Spam Detection System Using Machine Learning Algorithms"
    )
    parser.add_argument("--all", action="store_true", help="Run complete 16-stage ML Life Cycle")
    parser.add_argument("--eda", action="store_true", help="Run Exploratory Data Analysis & generate graphs")
    parser.add_argument("--preprocess", action="store_true", help="Run text preprocessing & train-test split")
    parser.add_argument("--train", action="store_true", help="Train and benchmark all 10 ML models")
    parser.add_argument("--tune", action="store_true", help="Run 5-fold stratified hyperparameter tuning")
    parser.add_argument("--test", action="store_true", help="Run automated pytest test suite")
    parser.add_argument("--predict", type=str, help="Predict single email text")

    args = parser.parse_args()

    if args.all:
        run_complete_ml_lifecycle()
    elif args.eda:
        run_eda()
    elif args.preprocess:
        preprocess_dataset()
        split_train_test()
    elif args.train:
        _, X_test, _, y_test = split_train_test()
        X_train, _, y_train, _ = split_train_test()
        records = train_all_models(X_train, y_train)
        eval_res = evaluate_all_models(records, X_test, y_test)
        run_model_comparison(eval_res)
    elif args.tune:
        X_train, _, y_train, _ = split_train_test()
        run_all_tuning(X_train, y_train)
    elif args.test:
        run_tests()
    elif args.predict:
        res = predict_email(args.predict)
        print(format_prediction_result(res))
    else:
        display_menu()


if __name__ == "__main__":
    main()
