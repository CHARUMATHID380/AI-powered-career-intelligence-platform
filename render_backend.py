"""
CareerCast — render_backend/app.py

Standalone Flask service that hosts the narrow Random Forest / XGBoost
resume classifier (narrow_model.py). This is the service `api/predict_narrow`
proxies to via the RENDER_NARROW_URL environment variable.

This file was missing from the uploaded project (only narrow_model.py and
text_extract.py were present) — added so the narrow classifier can be run
locally, on the same pattern the code's own comments describe.

RUN LOCALLY:
    python render_backend/app.py
    -> serves http://127.0.0.1:5001/api/predict_narrow
Then set RENDER_NARROW_URL=http://127.0.0.1:5001/api/predict_narrow
before starting dev_server.py so /api/predict_narrow proxies here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request
from flask_cors import CORS

from narrow_model import predict_narrow, NARROW_READY, NARROW_CATEGORIES
from text_extract import extract_text, ALLOWED_EXTENSIONS

MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)


@app.route("/api/predict_narrow", methods=["POST"])
def predict_narrow_route():
    if not NARROW_READY:
        return jsonify({"error": "Narrow classifier model files failed to load."}), 500

    model = request.form.get("model", "ensemble")

    if "resume_file" in request.files and request.files["resume_file"].filename:
        f = request.files["resume_file"]
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: .{ext}"}), 400
        try:
            resume_text = extract_text(f.filename, f.stream)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Could not read file: {exc}"}), 400
    else:
        resume_text = request.form.get("resume_text", "")

    if not resume_text or not resume_text.strip():
        return jsonify({"error": "Paste resume text or upload a PDF/DOCX/TXT file."}), 400

    results = predict_narrow(resume_text, model=model)
    return jsonify({
        "results": results,
        "model": model,
        "categories": NARROW_CATEGORIES,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)