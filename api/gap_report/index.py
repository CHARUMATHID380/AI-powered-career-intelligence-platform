"""
CareerCast — /api/gap_report
Full skill gap report: missing/present skills, resume wording suggestions, courses.
Ported from fastapi_service/main.py's /gap-report endpoint.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
from clean_text import extract_skills  # noqa: E402
from skill_gap import JOB_SKILLS, generate_gap_report  # noqa: E402

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/api/gap_report", methods=["POST"])
def gap_report():
    data = request.get_json(silent=True) or {}
    resume_text = data.get("resume_text", "")
    job_category = data.get("job_category", "")

    if not resume_text or not resume_text.strip():
        return jsonify({"error": "resume_text is required."}), 400
    if job_category not in JOB_SKILLS:
        return jsonify({"error": f"Unknown job category. Known: {list(JOB_SKILLS)}"}), 400

    skills = extract_skills(resume_text)
    result = generate_gap_report(resume_text, skills, job_category)
    result["extracted_skills"] = skills
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)