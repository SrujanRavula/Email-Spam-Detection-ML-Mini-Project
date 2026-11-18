# Demonstration Video Recording Script & Narration Guide
## Email Spam Detection System Using Machine Learning Algorithms

> **Video Status:** Recording Scheduled / Demonstration Script Active  
> **Target Video File:** `demo/Email_Spam_Detection_Demo.mp4`  
> **Estimated Video Duration:** 3 minutes 45 seconds  
> **Resolution:** 1920x1080 (1080p, 60fps)  
> **Audio:** Clear English technical walkthrough  

---

## Recording Sequence & Timecodes

### [00:00 - 00:30] Scene 1: Project Overview & Architecture
- **Visual:** Open VS Code showing `Email_Spam_Detection_ML` project root. Expand file explorer displaying `src/`, `data/`, `models/`, `results/`, `tests/`.
- **Narration:**
  > *"Hello and welcome to this demonstration of the Email Spam Detection System implemented using Scikit-learn, NLTK, and XGBoost following the complete 16-stage Machine Learning Life Cycle. In this project, we benchmarked ten supervised machine learning algorithms across probabilistic, linear, tree-based, nearest-neighbor, and boosting families on an authentic 5,728 email corpus derived from Apache SpamAssassin and Enron."*

### [00:30 - 01:00] Scene 2: Environment Verification & Automated Testing
- **Visual:** Open integrated PowerShell terminal in VS Code. Activate virtual environment and run automated pytest test suite.
- **Command Executed:**
  ```powershell
  .venv\Scripts\Activate.ps1
  pytest tests/ -v
  ```
- **Narration:**
  > *"First, let's verify system integrity using our automated Pytest suite. We run seventeen independent unit and integration tests covering data loading, schema validation, domain-aware tokenization, stopword preservation, data leakage prevention, matrix sparsity verification, model serialization, and live spam/ham predictions. As you can see, all seventeen tests pass with 100% success."*

### [01:00 - 01:45] Scene 3: Exploratory Data Analysis & Visualizations
- **Visual:** Display `results/graphs/` folder in VS Code. Open `figure_5_1_class_distribution.png`, `figure_5_2_length_distributions.png`, and `figure_5_4_top_vocabulary.png`.
- **Command Executed:**
  ```powershell
  python main.py --eda
  ```
- **Narration:**
  > *"Next, we inspect the Exploratory Data Analysis. Figure 5.1 illustrates the natural class imbalance: 75.98% legitimate emails versus 24.02% spam. Figure 5.2 demonstrates character and word density distributions, revealing that legitimate emails exhibit a longer tail with an average of 1,634 characters compared to 1,317 characters in spam. Figure 5.4 displays distinctive vocabulary words, where spam prominently features tokens like dollartoken, numbertoken, exclaimtoken, free, and click."*

### [01:45 - 02:30] Scene 4: 10-Algorithm Comparative Benchmark
- **Visual:** Run model training and display `results/model_comparison.txt` and `figure_12_1_multi_metric_comparison.png`.
- **Command Executed:**
  ```powershell
  python main.py --train
  ```
- **Narration:**
  > *"Now we examine the comparative performance across all ten implemented algorithms evaluated on the 1,139 untouched test emails. Linear models strongly outperformed tree ensembles in this 5,000-dimensional sparse space. Calibrated Support Vector Machine and Logistic Regression achieved the highest F1-scores of 0.9854 and 0.9759. Tree-based models like Decision Trees struggled with sparse word combinations, while Bernoulli Naive Bayes suffered from 72 false positives."*

### [02:30 - 03:00] Scene 5: Hyperparameter Tuning & Best Model Export
- **Visual:** Show hyperparameter tuning output and open `results/confusion_matrices/figure_14_1_best_svm_confusion_matrix.png`.
- **Command Executed:**
  ```powershell
  python main.py --tune
  ```
- **Narration:**
  > *"Using 5-fold stratified cross-validation on training data exclusively, we tuned the leading candidate models. For Support Vector Machine, regularization C=5.0 with unigram TF-IDF yielded the highest cross-validation score of 0.9853. Evaluated on the untouched test partition, our tuned Calibrated LinearSVC achieved 99.56% overall accuracy, 99.27% spam precision, 98.91% spam recall, and a 0.9909 F1-score with only 2 false positives out of 865 legitimate emails."*

### [03:00 - 03:45] Scene 6: Interactive Terminal Interface & Live Predictions
- **Visual:** Run interactive CLI and test live emails.
- **Command Executed:**
  ```powershell
  python main.py
  # Select option 1 and enter spam email
  Subject: Urgent Winner Notification! You have won $500,000 cash in our international lottery! Claim here now: http://lottery-claim.com
  # Select option 1 and enter ham email
  Subject: Machine Learning Seminar Schedule. Hello students, our next lecture on Support Vector Machines is scheduled for Friday at 10 AM in Room 302.
  ```
- **Narration:**
  > *"Finally, we launch our interactive terminal application. We enter a high-risk financial lottery message. The pipeline normalizes the text and instantly renders the verdict: SPAM DETECTED with 98.73% confidence. Next, we test a legitimate university lecture schedule. The system immediately classifies it as LEGITIMATE HAM with 91.56% confidence. The sub-millisecond inference time of 0.16 milliseconds per message makes this model suitable for real-time mail transfer agents."*

---

## Checklist for Video Presenter
- [ ] Ensure Windows terminal colors are high contrast (black background, readable font).
- [ ] Maximize VS Code terminal font size (`Ctrl + +`) during screen capture.
- [ ] Save recorded video as `demo/Email_Spam_Detection_Demo.mp4`.
- [ ] If uploaded to YouTube or Google Drive, insert the link into `README.md`.
