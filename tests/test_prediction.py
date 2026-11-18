"""
Test Suite for Model Loading and Prediction Inference
Validates joblib unpickling, spam/ham classification, probability outputs,
and edge case exception handling.
Follows Chapter 18 and Table 18.1 of the project technical specifications.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.prediction import load_best_model, predict_email


def test_invalid_input_handling():
    """Verifies that empty or whitespace strings raise ValueError gracefully."""
    with pytest.raises(ValueError):
        predict_email("")

    with pytest.raises(ValueError):
        predict_email("    \n\t   ")


def test_model_loading():
    """Verifies that the serialized best model unpickles cleanly as an active Pipeline."""
    model = load_best_model()
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_spam_prediction():
    """Verifies that an aggressive lottery/prize spam email is classified as SPAM with high confidence."""
    lottery_spam = (
        "Subject: Urgent Winner Notification! You have won $500,000 cash in our international "
        "lottery! Claim here now: http://lottery-claim.com"
    )
    result = predict_email(lottery_spam)
    assert result["prediction"] == config.SPAM_LABEL
    assert result["verdict"] == "[SPAM DETECTED]"
    assert result["confidence"] > 80.0
    assert result["spam_probability"] > 80.0


def test_ham_prediction():
    """Verifies that a legitimate academic seminar email is classified as HAM with high confidence."""
    academic_ham = (
        "Subject: Machine Learning Seminar Schedule. Hello students, our next lecture on "
        "Support Vector Machines is scheduled for Friday at 10 AM in Room 302. Please read chapter 5 beforehand."
    )
    result = predict_email(academic_ham)
    assert result["prediction"] == config.HAM_LABEL
    assert result["verdict"] == "[LEGITIMATE HAM]"
    assert result["confidence"] > 80.0
    assert result["ham_probability"] > 80.0


def test_prediction_output_structure():
    """Verifies the keys and types in the returned inference dictionary."""
    sample = "Meeting rescheduled to 3 PM tomorrow. Please bring the quarterly report."
    res = predict_email(sample)
    expected_keys = {
        "original_text",
        "cleaned_text",
        "prediction",
        "verdict",
        "confidence",
        "spam_probability",
        "ham_probability",
    }
    assert expected_keys.issubset(res.keys())
    assert isinstance(res["prediction"], int)
    assert isinstance(res["confidence"], float)
    assert abs((res["spam_probability"] + res["ham_probability"]) - 100.0) < 0.01
