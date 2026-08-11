"""
Lightweight readiness check for the narrow (RF/XGBoost) classifier.

Deliberately does NOT import narrow_model.py (which imports xgboost) —
this file only checks that the expected artifact files exist on disk.
Used by api/index/index.py so the home page never pulls xgboost into
its own function bundle.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_FILES = [
    "rf_model.joblib",
    "xgb_model.json",
    "tfidf_vectorizer.joblib",
    "label_encoder.joblib",
]

NARROW_CATEGORIES = ["Java Developer", "Business Analyst", "Project Manager"]


def narrow_is_ready() -> bool:
    return all(os.path.exists(os.path.join(BASE_DIR, f)) for f in REQUIRED_FILES)
