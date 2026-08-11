"""
CareerCast — /api/predict_narrow
Random Forest / XGBoost classifier, trained on a smaller hand-labeled set,
recognizes only 3 categories (Java Developer, Business Analyst, Project
Manager). This is the ONLY function in the app allowed to import
narrow_model.py — keeping the xgboost dependency isolated here is the
whole point of the split, so don't import this module from api/index or
api/predict.
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
from text_extract import extract_text, ALLOWED_EXTENSIONS  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import narrow_model  # noqa: E402

from flask import Flask, jsonify, request

MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/api/predict_narrow", methods=["POST"])
def predict_narrow_route():
    if not narrow_model.NARROW_READY:
        return jsonify({
            "error": "Narrow classifier not loaded. Check that rf_model.joblib, xgb_model.json, "
                     "tfidf_vectorizer.joblib, and label_encoder.joblib sit next to app.py."
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

    model_choice = request.form.get("model", "ensemble")  # "rf" | "xgb" | "ensemble"
    if model_choice not in ("rf", "xgb", "ensemble"):
        model_choice = "ensemble"

    try:
        results = narrow_model.predict_narrow(resume_text, model=model_choice, n=3)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    if not results:
        return jsonify({"error": "Resume text was too short or empty after cleaning."}), 400

    return jsonify({
        "results": results,
        "model": model_choice,
        "categories": narrow_model.NARROW_CATEGORIES,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
