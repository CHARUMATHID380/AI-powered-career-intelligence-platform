"""
CareerCast — /api/cohort
Analyse multiple resumes in one request and return aggregate statistics:
category distribution, top skills frequency, average confidence, and
per-resume results.
"""

import io
import os
import sys
from collections import Counter

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
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024   # 32 MB total for batch

_model       = None
_model_error = None
try:
    _model = joblib.load(MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    _model_error = str(exc)


def _predict_top(resume_text: str, n: int = 1):
    cleaned = clean_resume_text(resume_text)
    if not cleaned:
        return []
    proba   = _model.predict_proba([cleaned])[0]
    classes = _model.classes_
    ranked  = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:n]
    return [{"job": job, "score": round(float(s) * 100, 2)} for job, s in ranked]


@app.route("/api/cohort", methods=["POST"])
def cohort():
    if _model is None:
        return jsonify({"error": "Model not loaded."}), 500

    files = request.files.getlist("resume_files")
    if not files:
        return jsonify({"error": "Upload at least one file."}), 400

    rows, errors = [], []

    for f in files:
        filename = f.filename or ""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            errors.append({"file": filename, "error": f"Unsupported type .{ext}"})
            continue
        try:
            raw  = extract_text(filename, io.BytesIO(f.read()))
            top  = _predict_top(raw, n=1)
            if not top:
                errors.append({"file": filename, "error": "Text too short after cleaning."})
                continue
            skills = extract_skills(raw)

            # Skill-gap suggestions for the top role (if the role is in taxonomy)
            top_role = top[0]["job"]
            gap_missing: list[str] = []
            if top_role in JOB_SKILLS:
                try:
                    gap = compute_gap(skills, top_role)
                    gap_missing = gap.missing_core[:4]
                except Exception:  # noqa: BLE001
                    pass

            rows.append({
                "file":       filename,
                "top_match":  top_role,
                "confidence": top[0]["score"],
                "skills":     skills,
                "gap_missing": gap_missing,
            })
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": filename, "error": str(exc)})

    if not rows and errors:
        return jsonify({"error": "All files failed.", "details": errors}), 400

    cat_counter   = Counter(r["top_match"] for r in rows)
    skill_counter = Counter(s for r in rows for s in r["skills"])
    avg_conf      = round(sum(r["confidence"] for r in rows) / len(rows), 1) if rows else 0.0

    return jsonify({
        "rows":             rows,
        "category_counts":  dict(cat_counter.most_common()),
        "top_skills":       dict(skill_counter.most_common(12)),
        "avg_confidence":   avg_conf,
        "total":            len(rows),
        "errors":           errors,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003)
