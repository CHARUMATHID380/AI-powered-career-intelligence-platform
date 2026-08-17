"""
CareerCast — Milestone 3 FastAPI Service

Three endpoints, per the milestone brief:
  POST /predict     — resume -> top 5 ranked job matches
  POST /recommend    — job category -> course recommendations only (quick)
  POST /gap-report   — resume + job category -> full skill gap report

This is a SEPARATE service from your existing Flask app (api/index,
api/predict, api/predict_narrow). Run it standalone during development;
deployment target (Vercel vs Render vs elsewhere) is a decision for later
depending on its dependency weight — see notes at the bottom of this file.

INTEGRATION TODOs (marked clearly below) — this file makes reasonable
assumptions about your existing code that you'll need to confirm/adjust:
  1. Broad classifier loading (`load_broad_model`) assumes
     resume_job_classifier.joblib is a scikit-learn Pipeline exposing
     .predict_proba() and .classes_. Adjust if your actual model is
     structured differently (e.g. separate vectorizer + classifier files).
  2. Skill extraction (`extract_skills`) is a SIMPLE FALLBACK — it just
     keyword-matches against the skill_gap taxonomy. Swap this out for
     your real, more sophisticated skill-extraction logic if you have one
     already built elsewhere in the project.
  3. Text cleaning/extraction reuses your existing lib/clean_text.py and
     lib/text_extract.py via the same sys.path pattern your other
     functions already use.
"""

from __future__ import annotations

import io
import os
import sys

import joblib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# --- Reuse existing shared modules (same pattern as api/predict_narrow) ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from clean_text import clean_text  # noqa: E402
from text_extract import extract_text, ALLOWED_EXTENSIONS  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from skill_gap import JOB_SKILLS, generate_gap_report, add_course_suggestions, GapReport  # noqa: E402

app = FastAPI(title="CareerCast Skill Gap & Prediction API", version="0.1.0")

# ---------------------------------------------------------------------------
# MODEL LOADING (broad classifier)
# ---------------------------------------------------------------------------
BROAD_MODEL_PATH = os.environ.get(
    "BROAD_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "resume_job_classifier.joblib"),
)

_broad_model = None
_broad_model_error = None


def load_broad_model():
    global _broad_model, _broad_model_error
    try:
        _broad_model = joblib.load(BROAD_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        _broad_model_error = str(exc)
        _broad_model = None


load_broad_model()


def predict_top5(resume_text: str) -> list[dict]:
    """
    TODO: confirm this matches your actual broad classifier's structure.
    Assumes `_broad_model` is a fitted sklearn Pipeline/Estimator with
    .predict_proba() and .classes_ (standard for TF-IDF + LogisticRegression
    resume classifiers).
    """
    if _broad_model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Broad classifier not loaded: {_broad_model_error}",
        )

    cleaned = clean_text(resume_text)
    probs = _broad_model.predict_proba([cleaned])[0]
    classes = _broad_model.classes_

    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:5]
    return [{"job_category": str(c), "confidence": round(float(p), 4)} for c, p in ranked]


# ---------------------------------------------------------------------------
# SKILL EXTRACTION (simple fallback — replace with your real extractor)
# ---------------------------------------------------------------------------
def extract_skills(resume_text: str) -> list[str]:
    """
    FALLBACK ONLY. Keyword-matches the resume text against every skill
    that appears anywhere in JOB_SKILLS. Replace this with your project's
    existing, more sophisticated skill-extraction logic if one exists —
    this is here so the endpoints are fully functional out of the box.
    """
    lowered = resume_text.lower()
    all_skills = set()
    for taxonomy in JOB_SKILLS.values():
        all_skills.update(taxonomy["core"])
        all_skills.update(taxonomy["recommended"])

    return [skill for skill in all_skills if skill in lowered]


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------------------------------
class PredictResponse(BaseModel):
    results: list[dict]


class RecommendRequest(BaseModel):
    job_category: str
    missing_skills: list[str] | None = None  # optional override


class GapReportRequest(BaseModel):
    resume_text: str
    job_category: str


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
async def _get_resume_text(resume_text: str | None, resume_file: UploadFile | None) -> str:
    if resume_file is not None and resume_file.filename:
        ext = resume_file.filename.rsplit(".", 1)[-1].lower() if "." in resume_file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type .{ext}. Use PDF, DOCX, or TXT.")
        file_bytes = await resume_file.read()
        text = extract_text(resume_file.filename, io.BytesIO(file_bytes))
    elif resume_text:
        text = resume_text
    else:
        raise HTTPException(400, "Paste resume text or upload a PDF/DOCX/TXT file.")

    if not text or not text.strip():
        raise HTTPException(400, "No readable text was found in the resume.")
    return text


@app.post("/predict", response_model=PredictResponse)
async def predict(
    resume_text: str | None = Form(None),
    resume_file: UploadFile | None = File(None),
):
    """Top 5 ranked job matches for a resume."""
    text = await _get_resume_text(resume_text, resume_file)
    return {"results": predict_top5(text)}


@app.post("/recommend")
async def recommend(payload: RecommendRequest):
    """Quick course recommendations only, for a given job category."""
    if payload.job_category not in JOB_SKILLS:
        raise HTTPException(400, f"Unknown job category. Known: {list(JOB_SKILLS)}")

    missing = payload.missing_skills or JOB_SKILLS[payload.job_category]["core"]
    report = GapReport(job_category=payload.job_category, missing_core=missing)
    add_course_suggestions(report)
    return {"job_category": payload.job_category, "courses": report.course_suggestions}


@app.post("/gap-report")
async def gap_report(payload: GapReportRequest):
    """Full skill gap report: missing/present skills, resume wording suggestions, and courses."""
    if payload.job_category not in JOB_SKILLS:
        raise HTTPException(400, f"Unknown job category. Known: {list(JOB_SKILLS)}")

    skills = extract_skills(payload.resume_text)
    result = generate_gap_report(payload.resume_text, skills, payload.job_category)
    result["extracted_skills"] = skills  # useful for debugging/UI display
    return result


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "broad_model_loaded": _broad_model is not None,
        "broad_model_error": _broad_model_error,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ---------------------------------------------------------------------------
# DEPLOYMENT NOTE
# ---------------------------------------------------------------------------
# This service needs: fastapi, uvicorn, joblib, scikit-learn (for the broad
# model), requests (for course search). That's lighter than the narrow
# classifier's xgboost stack, but still has scikit-learn in it — so before
# deploying, decide: run this on Render alongside (or merged into) your
# existing render_backend/app.py, OR keep it separate from Vercel entirely
# to avoid reintroducing the bundle-size problem you just fixed.
