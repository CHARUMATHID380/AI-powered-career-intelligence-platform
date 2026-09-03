"""
CareerCast — /api/compare
Side-by-side comparison of two resumes.  Accepts either pasted text
(form fields text_a / text_b) or uploaded files (file_a / file_b).
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
from clean_text import clean_resume_text, extract_skills, nltk_ready  # noqa: E402
from text_extract import extract_text, ALLOWED_EXTENSIONS              # noqa: E402
from skill_gap import JOB_SKILLS, compute_gap                         # noqa: E402

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_PATH = os.path.join(BASE_DIR, "resume_job_classifier.joblib")

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB

_model       = None
_model_error = None
try:
    _model = joblib.load(MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    _model_error = str(exc)


def _predict(resume_text: str, n: int = 5):
    cleaned = clean_resume_text(resume_text)
    if not cleaned:
        return []
    proba   = _model.predict_proba([cleaned])[0]
    classes = _model.classes_
    ranked  = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:n]
    return [{"job": job, "score": round(float(s) * 100, 2)} for job, s in ranked]


def _resolve(text_field: str, file_field: str):
    """Return (raw_text, label) from either form text or uploaded file."""
    if request.files.get(file_field) and request.files[file_field].filename:
        f  = request.files[file_field]
        fn = f.filename
        ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported type .{ext}")
        return extract_text(fn, io.BytesIO(f.read())), fn
    text = (request.form.get(text_field) or "").strip()
    if not text:
        raise ValueError("No text or file provided.")
    return text, "Candidate"


def _analyse(raw_text: str, label: str) -> dict:
    results = _predict(raw_text, n=5)
    if not results:
        raise ValueError(f"Text too short after cleaning for {label}.")
    skills    = extract_skills(raw_text)
    top_role  = results[0]["job"]

    # Skill-gap for top role
    gap_missing: list[str] = []
    if top_role in JOB_SKILLS:
        try:
            gap         = compute_gap(skills, top_role)
            gap_missing = gap.missing_core[:6]
        except Exception:  # noqa: BLE001
            pass

    return {
        "label":            label,
        "top_match":        top_role,
        "top_score":        results[0]["score"],
        "ranked_jobs":      results,
        "skills":           skills,
        "skills_suggested": gap_missing,
    }


@app.route("/api/compare", methods=["POST"])
def compare():
    if _model is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        text_a, label_a = _resolve("text_a", "file_a")
        text_b, label_b = _resolve("text_b", "file_b")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        a = _analyse(text_a, label_a)
        b = _analyse(text_b, label_b)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    winner = "a" if a["top_score"] >= b["top_score"] else "b"
    return jsonify({"a": a, "b": b, "winner": winner})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5004)
