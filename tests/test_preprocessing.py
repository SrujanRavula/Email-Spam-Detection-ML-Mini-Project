"""
Test Suite for Data Preprocessing
Validates text cleaning, normalization, token replacement, and edge cases.
Follows Chapter 18 and Table 18.1 of the project technical specifications.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import clean_email_text


def test_clean_text_lowercase():
    """Verifies that all uppercase characters are transformed to lowercase."""
    raw = "Subject: THIS IS AN URGENT ALL CAPS MESSAGE"
    cleaned = clean_email_text(raw)
    assert cleaned == cleaned.lower()
    assert "urgent" in cleaned


def test_html_stripping():
    """Verifies that HTML tags (<p>, <a>, <html>, etc.) are stripped."""
    raw = "Subject: <html><body><h1>Special Offer!</h1><p>Click <a href='http://x.com'>here</a></p></body></html>"
    cleaned = clean_email_text(raw)
    assert "<html>" not in cleaned
    assert "<body>" not in cleaned
    assert "<h1>" not in cleaned
    assert "<p>" not in cleaned
    assert "special" in cleaned
    assert "offer" in cleaned


def test_url_normalization():
    """Verifies that web URLs are normalized to 'httpurl' token."""
    raw = "Visit https://secure-bank-login.com/verify or www.lottery-winners.org today"
    cleaned = clean_email_text(raw)
    assert "httpurl" in cleaned
    assert "https://secure-bank-login.com" not in cleaned
    assert "www.lottery-winners.org" not in cleaned


def test_number_and_currency_tokens():
    """Verifies that currency symbols map to 'dollartoken' and numbers map to 'numbertoken'."""
    raw = "You have won $1,000,000 in cash award 500"
    cleaned = clean_email_text(raw)
    assert "dollartoken" in cleaned
    assert "numbertoken" in cleaned
    assert "$1,000,000" not in cleaned


def test_exclamation_normalization():
    """Verifies that urgent exclamations (!!+) map to 'exclaimtoken'."""
    raw = "URGENT! Claim your cash prize immediately!!!"
    cleaned = clean_email_text(raw)
    assert "exclaimtoken" in cleaned


def test_empty_or_whitespace():
    """Verifies that empty string or whitespace returns safe empty string without exception."""
    assert clean_email_text("") == ""
    assert clean_email_text("   \t\n   ") == ""
    assert clean_email_text(None) == ""


def test_stopword_preservation():
    """Verifies that critical spam indicators ('not', 'no', 'free', 'win', 'won') are preserved."""
    raw = "This is not a scam you can win free prizes and won reward"
    cleaned = clean_email_text(raw)
    tokens = cleaned.split()
    assert "not" in tokens
    assert "free" in tokens
    assert "win" in tokens or "won" in tokens
