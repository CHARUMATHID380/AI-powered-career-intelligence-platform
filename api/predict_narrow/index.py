"""
CareerCast — /api/predict_narrow  (Vercel side)

This function no longer runs the Random Forest / XGBoost models itself.
It is a thin proxy: it forwards the incoming request to the standalone
Flask service deployed on Render (see render_backend/app.py), which is
the only place narrow_model.py / xgboost / scikit-learn get imported.

Keeping this file free of ML imports is the whole point of the split —
do not import narrow_model here, and do not add joblib/xgboost/sklearn
back to this function's requirements.txt.
"""

import os

import requests
from flask import Flask, jsonify, request

MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
REQUEST_TIMEOUT = 30  # seconds — model inference on Render's free tier can be slow to cold-start

# Set this in Vercel's Environment Variables (Project Settings -> Environment Variables),
# e.g. RENDER_NARROW_URL = https://careercast-narrow.onrender.com/api/predict_narrow
RENDER_NARROW_URL = os.environ.get("RENDER_NARROW_URL", "")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/api/predict_narrow", methods=["POST"])
def predict_narrow_route():
    if not RENDER_NARROW_URL:
        return jsonify({
            "error": "RENDER_NARROW_URL is not configured. Set it in Vercel's environment variables "
                     "to point at your Render service, e.g. https://<your-app>.onrender.com/api/predict_narrow"
        }), 500

    try:
        if "resume_file" in request.files and request.files["resume_file"].filename:
            f = request.files["resume_file"]
            files = {"resume_file": (f.filename, f.read(), f.mimetype)}
            data = {"model": request.form.get("model", "ensemble")}
            resp = requests.post(RENDER_NARROW_URL, files=files, data=data, timeout=REQUEST_TIMEOUT)
        elif request.form.get("resume_text"):
            data = {
                "resume_text": request.form.get("resume_text", ""),
                "model": request.form.get("model", "ensemble"),
            }
            resp = requests.post(RENDER_NARROW_URL, data=data, timeout=REQUEST_TIMEOUT)
        else:
            return jsonify({"error": "Paste resume text or upload a PDF/DOCX/TXT file."}), 400
    except requests.exceptions.Timeout:
        return jsonify({"error": "The classifier service timed out. It may be cold-starting — try again in a moment."}), 504
    except requests.exceptions.RequestException as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not reach the classifier service: {exc}"}), 502

    # Pass through the Render service's response (status code + JSON body) as-is
    try:
        return jsonify(resp.json()), resp.status_code
    except ValueError:
        return jsonify({"error": "Classifier service returned an invalid response."}), 502


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
