"""
CareerCast — home route only.
Deliberately dependency-light: no scikit-learn, no xgboost, no nltk.
Just renders the landing page and reports whether each model's files
are present on disk, without loading either model into memory.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
from narrow_status import narrow_is_ready, NARROW_CATEGORIES  # noqa: E402

from flask import Flask, render_template

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)


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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
