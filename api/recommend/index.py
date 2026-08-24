"""
CareerCast — /api/recommend
Quick course recommendations for a given job category (no resume needed).
Ported from fastapi_service/main.py's /recommend endpoint.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
from skill_gap import JOB_SKILLS, GapReport, add_course_suggestions  # noqa: E402

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True) or {}
    job_category = data.get("job_category", "")
    missing_skills = data.get("missing_skills")

    if job_category not in JOB_SKILLS:
        return jsonify({"error": f"Unknown job category. Known: {list(JOB_SKILLS)}"}), 400

    missing = missing_skills or JOB_SKILLS[job_category]["core"]
    report = GapReport(job_category=job_category, missing_core=missing)
    add_course_suggestions(report)
    return jsonify({"job_category": job_category, "courses": report.course_suggestions})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003)