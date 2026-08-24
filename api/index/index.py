"""
CareerCast — page routes + data API endpoints.

Page routes (GET, returns HTML):
  /              → landing.html   (unchanged)
  /tool          → index.html     (unchanged)
  /dashboard     → results_dashboard.html
  /courses       → courses.html   (passes categories + job_skills to template)
  /skill-gap     → skill_gap.html (passes categories to template)
  /explore       → explore.html   (passes job_skills, totals to template)

Data API endpoints (GET, returns JSON):
  /api/categories   → sorted list of all job category names
  /api/job_skills   → full {category: {core: [...], recommended: [...]}} dict
  /api/job_skills/<category> → single category's skill entry

Deliberately dependency-light for the page routes: no scikit-learn, no
xgboost, no nltk — only the skill taxonomy from lib/skill_gap.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
from narrow_status import narrow_is_ready, NARROW_CATEGORIES  # noqa: E402
from skill_gap import JOB_SKILLS  # noqa: E402

from flask import Flask, jsonify, render_template

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Pre-compute derived values used by multiple routes
_CATEGORIES = sorted(JOB_SKILLS.keys())
_TOTAL_SKILLS = sum(
    len(v.get("core", [])) + len(v.get("recommended", []))
    for v in JOB_SKILLS.values()
)


# ---------------------------------------------------------------------------
# Existing page routes (UNCHANGED)
# ---------------------------------------------------------------------------

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/tool")
def index():
    model_path = os.path.join(BASE_DIR, "resume_job_classifier.joblib")
    model_ready = os.path.exists(model_path)
    return render_template(
        "index.html",
        model_ready=model_ready,
        model_error=None if model_ready else "resume_job_classifier.joblib not found.",
        narrow_ready=narrow_is_ready(),
        narrow_categories=NARROW_CATEGORIES,
    )


# ---------------------------------------------------------------------------
# New page routes
# ---------------------------------------------------------------------------

@app.route("/dashboard")
def dashboard():
    """Interactive results dashboard with Chart.js visualisations."""
    return render_template("results_dashboard.html")


@app.route("/courses")
def courses():
    """Course recommendations page — browse by job category and skill."""
    return render_template(
        "courses.html",
        categories=_CATEGORIES,
        job_skills=JOB_SKILLS,
    )


@app.route("/skill-gap")
def skill_gap():
    """Dedicated skill gap analysis page with visual charts."""
    return render_template(
        "skill_gap.html",
        categories=_CATEGORIES,
    )


@app.route("/explore")
def explore():
    """Career explorer — browse all job categories and their skill maps."""
    return render_template(
        "explore.html",
        job_skills=JOB_SKILLS,
        categories=_CATEGORIES,
        total_categories=len(_CATEGORIES),
        total_skills=_TOTAL_SKILLS,
    )


@app.route("/app")
def spa():
    """Single-page application — all tools in one unified view."""
    model_path = os.path.join(BASE_DIR, "resume_job_classifier.joblib")
    model_ready = os.path.exists(model_path)
    return render_template(
        "app.html",
        model_ready=model_ready,
        model_error=None if model_ready else "resume_job_classifier.joblib not found.",
        narrow_ready=narrow_is_ready(),
        narrow_categories=NARROW_CATEGORIES,
        categories=_CATEGORIES,
        job_skills=JOB_SKILLS,
        total_categories=len(_CATEGORIES),
        total_skills=_TOTAL_SKILLS,
    )


# ---------------------------------------------------------------------------
# JSON data endpoints (used by the new pages' JS and by external callers)
# ---------------------------------------------------------------------------

@app.route("/api/categories", methods=["GET"])
def api_categories():
    """Return the sorted list of all tracked job categories."""
    return jsonify({"categories": _CATEGORIES, "total": len(_CATEGORIES)})


@app.route("/api/job_skills", methods=["GET"])
def api_job_skills():
    """Return the complete skill taxonomy as JSON."""
    return jsonify(JOB_SKILLS)


@app.route("/api/job_skills/<path:category>", methods=["GET"])
def api_job_skills_single(category):
    """Return core and recommended skills for one job category."""
    if category not in JOB_SKILLS:
        return jsonify({"error": f"Unknown category '{category}'."}), 404
    return jsonify({
        "job_category": category,
        "core": JOB_SKILLS[category].get("core", []),
        "recommended": JOB_SKILLS[category].get("recommended", []),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
