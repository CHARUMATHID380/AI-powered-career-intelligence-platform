"""
CareerCast — Master Flask application
======================================
Entry point for both local development and Render deployment.

Local:   python dev_server.py
Render:  gunicorn dev_server:app
"""

import importlib.util
import os
import sys

# ── Startup: download NLTK data if not already present ──────────────────────
import nltk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")

# Always add local nltk_data to search path
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

# On Render (and other cloud hosts) /tmp is writable — download there as fallback
TMP_NLTK = "/tmp/nltk_data"
if TMP_NLTK not in nltk.data.path:
    nltk.data.path.insert(0, TMP_NLTK)

def _ensure_nltk():
    needed = ["stopwords", "wordnet", "omw-1.4"]
    for pkg in needed:
        try:
            if pkg == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif pkg == "wordnet":
                nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)
            nltk.download(pkg, download_dir=TMP_NLTK, quiet=True)

_ensure_nltk()

# ── Add root to sys.path ─────────────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)

from flask import Flask

# ── Submodule loader ─────────────────────────────────────────────────────────
def load_submodule(name, rel_path):
    path = os.path.join(BASE_DIR, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ── Master app ───────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
)

SUBMODULES = [
    ("careercast_index",          "api/index/index.py"),
    ("careercast_predict",        "api/predict/index.py"),
    ("careercast_predict_narrow", "api/predict_narrow/index.py"),
    ("careercast_gap_report",     "api/gap_report/index.py"),
    ("careercast_recommend",      "api/recommend/index.py"),
    # Milestone 4
    ("careercast_cohort",         "api/cohort/index.py"),
    ("careercast_compare",        "api/compare/index.py"),
    ("careercast_pdf_report",     "api/pdf_report/index.py"),
]

for mod_name, rel_path in SUBMODULES:
    mod = load_submodule(mod_name, rel_path)
    sub_app = mod.app
    for rule in sub_app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        view_func = sub_app.view_functions[rule.endpoint]
        app.add_url_rule(
            rule.rule,
            endpoint=f"{mod_name}.{rule.endpoint}",
            view_func=view_func,
            methods=rule.methods,
        )

if __name__ == "__main__":
    # Local development only — Render uses gunicorn
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
