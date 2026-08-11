"""
narrow_model.py — Secondary Random Forest / XGBoost resume classifier
=======================================================================
Separate, clearly-scoped classifier trained on a smaller hand-labeled
set. Recognizes only 3 categories (Java Developer, Business Analyst,
Project Manager). Kept isolated from the broad Logistic Regression
model (resume_job_classifier.joblib) so results are never mixed.

RUN LOCALLY:
    1. Make sure rf_model.joblib, xgb_model.joblib, vectorizer.joblib,
       and label_encoder.joblib sit next to this file (produced by
       whatever script trained them in milestone2).
    2. app.py imports this module and exposes /api/predict_narrow.
"""

import os
import joblib
import numpy as np
from xgboost import XGBClassifier

# -----------------------------------------------------------------------
# Base directory of this file — same pattern as app.py, so paths work
# regardless of the working directory the app is launched from.
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_MODEL_PATH = os.path.join(BASE_DIR, "rf_model.joblib")
XGB_MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.json")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.joblib")

NARROW_CATEGORIES = ["Java Developer", "Business Analyst", "Project Manager"]

_rf_model = None
_xgb_model = None
_vectorizer = None
_label_encoder = None
NARROW_READY = False
_narrow_error = None

try:
    _rf_model = joblib.load(RF_MODEL_PATH)
    _vectorizer = joblib.load(VECTORIZER_PATH)
    _label_encoder = joblib.load(LABEL_ENCODER_PATH)

    # xgb_model.json was saved with XGBoost's native save_model(), so it
    # must be loaded via XGBClassifier().load_model(), not joblib.load().
    _xgb_model = XGBClassifier()
    _xgb_model.load_model(XGB_MODEL_PATH)

    NARROW_READY = True
except Exception as exc:  # noqa: BLE001
    _narrow_error = str(exc)
    NARROW_READY = False


def _aligned_proba(model, X):
    """
    Return this model's predicted probabilities re-ordered to match
    NARROW_CATEGORIES, regardless of the model's internal class order.
    Prevents RF and XGB from being blended against mismatched labels.
    """
    proba = model.predict_proba(X)[0]
    labels = list(_label_encoder.inverse_transform(model.classes_))
    return np.array([proba[labels.index(c)] for c in NARROW_CATEGORIES])


def predict_narrow(resume_text: str, model: str = "ensemble", n: int = 3):
    """
    Predict top-n job categories for resume_text using the narrow
    classifier. model: "rf" | "xgb" | "ensemble" (average of both).
    """
    if not NARROW_READY:
        raise RuntimeError(f"Narrow classifier not loaded: {_narrow_error}")

    cleaned = resume_text.strip()
    if not cleaned:
        return []

    X = _vectorizer.transform([cleaned])

    if model == "rf":
        proba = _aligned_proba(_rf_model, X)
    elif model == "xgb":
        proba = _aligned_proba(_xgb_model, X)
    else:
        proba = (_aligned_proba(_rf_model, X) + _aligned_proba(_xgb_model, X)) / 2

    ranked_idx = np.argsort(proba)[::-1][:n]
    return [
        {"role": NARROW_CATEGORIES[i], "confidence": round(float(proba[i]), 4)}
        for i in ranked_idx
    ]
