import importlib.util
import os
import sys

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from flask import Flask

# Each api/<name>/index.py is written as a standalone Vercel Python
# function (its own `app = Flask(__name__)` with the full route already
# declared, e.g. @app.route("/api/predict", ...)). There's no `handler`
# attribute and no __init__.py making `api.predict` importable as a
# package, so we load each index.py directly off disk and copy its
# routes onto one master Flask app instead.


def load_submodule(name, rel_path):
    path = os.path.join(BASE_DIR, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Master app — serves static/ from project root so new pages can
# reference /static/... assets if needed in the future.
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
)

SUBMODULES = [
    ("careercast_index",          "api/index/index.py"),           # page routes + data API
    ("careercast_predict",        "api/predict/index.py"),
    ("careercast_predict_narrow", "api/predict_narrow/index.py"),
    ("careercast_gap_report",     "api/gap_report/index.py"),
    ("careercast_recommend",      "api/recommend/index.py"),
    # ── Milestone 4 additions ──────────────────────────────────────
    ("careercast_cohort",         "api/cohort/index.py"),          # batch cohort analysis
    ("careercast_compare",        "api/compare/index.py"),         # side-by-side comparison
    ("careercast_pdf_report",     "api/pdf_report/index.py"),      # PDF export
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

if __name__ == '__main__':
    app.run(port=5001, debug=True)