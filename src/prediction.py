"""
Email Prediction Module
Loads the serialized winning pipeline and provides real-time inference
with calibrated probabilities, confidence scores, and formatted terminal outputs.
Follows Chapter 16 and Figures 16.1, 16.2, 16.3 of the project technical specifications.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils import setup_logger
from src.data_preprocessing import clean_email_text

logger = setup_logger("Prediction")

# Global pipeline cache
_cached_model = None


def load_best_model(model_path: Path = config.BEST_MODEL_PATH):
    """
    Loads and caches the serialized best model pipeline.
    Ensures inference runs in sub-millisecond time without retraining.
    """
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not model_path.exists():
        raise FileNotFoundError(
            f"Serialized model file not found at '{model_path}'. "
            "Please run model training or training pipeline first via 'python main.py --train'."
        )

    logger.info(f"Loading trained model from {model_path}...")
    _cached_model = joblib.load(model_path)
    return _cached_model


def predict_email(email_text: str, model=None) -> Dict[str, Any]:
    """
    Performs real-time inference on arbitrary raw email text.

    Returns:
        dict: {
            'original_text': str,
            'cleaned_text': str,
            'prediction': int (0 for Ham, 1 for Spam),
            'verdict': str,
            'confidence': float,
            'spam_probability': float,
            'ham_probability': float
        }
    """
    if not isinstance(email_text, str) or not email_text.strip():
        raise ValueError("Invalid input: Email text cannot be empty or solely whitespace.")

    if model is None:
        model = load_best_model()

    # Preprocess incoming raw email string
    cleaned = clean_email_text(email_text)
    if not cleaned.strip():
        cleaned = "emptytoken"

    # Inference via Pipeline (TfidfVectorizer -> CalibratedClassifierCV)
    y_pred = int(model.predict([cleaned])[0])
    probabilities = model.predict_proba([cleaned])[0]

    ham_prob = float(probabilities[config.HAM_LABEL])
    spam_prob = float(probabilities[config.SPAM_LABEL])

    verdict = "[SPAM DETECTED]" if y_pred == config.SPAM_LABEL else "[LEGITIMATE HAM]"
    confidence = (spam_prob if y_pred == config.SPAM_LABEL else ham_prob) * 100.0

    return {
        "original_text": email_text,
        "cleaned_text": cleaned,
        "prediction": y_pred,
        "verdict": verdict,
        "confidence": confidence,
        "spam_probability": spam_prob * 100.0,
        "ham_probability": ham_prob * 100.0,
    }


def format_prediction_result(res: Dict[str, Any], snippet_len: int = 80) -> str:
    """Formats inference result for terminal display matching Figure 16.2 / 16.3."""
    orig_snippet = res["original_text"][:snippet_len].replace("\n", " ")
    if len(res["original_text"]) > snippet_len:
        orig_snippet += "..."

    clean_snippet = res["cleaned_text"][:snippet_len]
    if len(res["cleaned_text"]) > snippet_len:
        clean_snippet += "..."

    output = [
        "",
        "==================================================================",
        "                   EMAIL CLASSIFICATION RESULT                    ",
        "==================================================================",
        f"Original Text Snippet : {orig_snippet}",
        f"Cleaned Text Sample   : {clean_snippet}",
        "------------------------------------------------------------------",
        f"VERDICT               : {res['verdict']}",
        f"CONFIDENCE            : {res['confidence']:.2f}%",
        f"Spam Probability      : {res['spam_probability']:.2f}%",
        f"Ham Probability       : {res['ham_probability']:.2f}%",
        "==================================================================",
        "",
    ]
    return "\n".join(output)


if __name__ == "__main__":
    sample_spam = "Subject: Urgent Winner Notification! You have won $500,000 cash in our international lottery! Claim here now: http://lottery-claim.com"
    try:
        res = predict_email(sample_spam)
        print(format_prediction_result(res))
    except Exception as e:
        print(f"Error: {e}")
