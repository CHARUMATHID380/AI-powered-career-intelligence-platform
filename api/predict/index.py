"""
CareerCast — /api/predict
Broad Logistic Regression resume classifier. Self-contained function:
does NOT import narrow_model.py, so xgboost never enters this bundle.
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
from clean_text import clean_resume_text, extract_skills, nltk_ready  # noqa: E402
from text_extract import extract_text, ALLOWED_EXTENSIONS  # noqa: E402

import joblib
from flask import Flask, jsonify, request

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_PATH = os.path.join(BASE_DIR, "resume_job_classifier.joblib")
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
TOP_N = 5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_model = None
_model_error = None
try:
    _model = joblib.load(MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    _model_error = str(exc)

if _model is not None and not nltk_ready():
    _model_error = "NLTK data failed to download at startup."


def predict_top_jobs(resume_text: str, n: int = TOP_N):
    cleaned = clean_resume_text(resume_text)
    if not cleaned:
        return []
    proba = _model.predict_proba([cleaned])[0]
    classes = _model.classes_
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:n]
    return [{"job": job, "score": round(float(score) * 100, 2)} for job, score in ranked]


@app.route("/api/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({
            "error": "Model not loaded. Place resume_job_classifier.joblib next to app.py and restart the server."
        }), 500

    resume_text = ""

    if "resume_file" in request.files and request.files["resume_file"].filename:
        f = request.files["resume_file"]
        filename = f.filename
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type .{ext}. Use PDF, DOCX, or TXT."}), 400
        try:
            file_bytes = f.read()
            resume_text = extract_text(filename, io.BytesIO(file_bytes))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Could not read file: {exc}"}), 400

    elif request.form.get("resume_text"):
        resume_text = request.form.get("resume_text", "")

    else:
        return jsonify({"error": "Paste resume text or upload a PDF/DOCX/TXT file."}), 400

    if not resume_text or not resume_text.strip():
        return jsonify({"error": "No readable text was found in the resume."}), 400

    try:
        results = predict_top_jobs(resume_text, n=TOP_N)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    if not results:
        return jsonify({"error": "Resume text was too short or empty after cleaning."}), 400

    preview = resume_text.strip().replace("\r", "")
    if len(preview) > 600:
        preview = preview[:600] + "…"

    skills = extract_skills(resume_text)

    return jsonify({"results": results, "preview": preview, "skills": skills})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
