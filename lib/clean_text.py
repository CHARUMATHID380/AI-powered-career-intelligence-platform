"""
Shared text-cleaning logic for the broad LR resume classifier.
MUST stay in sync with train_model.py, otherwise predictions will be
inaccurate. Only imported by api/predict/index.py — keep it free of
xgboost / narrow-model imports so it doesn't drag that weight into
other functions.
"""

import os
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

# Vercel: only /tmp is writable, so fall back there if the bundled data
# directory itself isn't writable/present.
TMP_NLTK_DIR = "/tmp/nltk_data"
if TMP_NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, TMP_NLTK_DIR)

_nltk_ready = True
try:
    stopwords.words("english")  # sanity check the bundled data loads
except Exception:  # noqa: BLE001
    _nltk_ready = False

if not _nltk_ready:
    try:
        nltk.download("stopwords", download_dir=TMP_NLTK_DIR, quiet=True)
        nltk.download("wordnet", download_dir=TMP_NLTK_DIR, quiet=True)
        nltk.download("omw-1.4", download_dir=TMP_NLTK_DIR, quiet=True)
        _nltk_ready = True
    except Exception:  # noqa: BLE001
        _nltk_ready = False

try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:  # noqa: BLE001
    STOPWORDS = set()
    _nltk_ready = False

LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"http\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def nltk_ready() -> bool:
    return _nltk_ready


def clean_resume_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = text.split()
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]
    text = " ".join(tokens)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


SKILL_KEYWORDS = [
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "SQL", "R",
    "HTML", "CSS", "PHP", "Go", "Ruby", "Swift", "Kotlin", "Scala",
    "Machine Learning", "Deep Learning", "Data Analysis", "Data Visualization",
    "Data Science", "Natural Language Processing", "Computer Vision",
    "Artificial Intelligence", "Statistics", "TensorFlow", "PyTorch",
    "Scikit-learn", "Pandas", "NumPy", "Keras",
    "React", "Angular", "Vue", "Node.js", "Django", "Flask", "Spring Boot",
    "REST API", "GraphQL", "Microservices",
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "DevOps",
    "CI/CD", "Git", "Linux",
    "Excel", "Power BI", "Tableau", "MySQL", "PostgreSQL", "MongoDB",
    "Project Management", "Agile", "Scrum", "Communication", "Leadership",
    "Teamwork", "Problem Solving", "Time Management",
    "Accounting", "Finance", "Marketing", "Sales", "SEO", "Content Writing",
    "Business Analysis", "Financial Modeling", "Auditing",
]
_SKILL_PATTERNS = [
    (skill, re.compile(r"(?<![a-zA-Z])" + re.escape(skill).replace(r"\ ", r"\s+") + r"(?![a-zA-Z])", re.IGNORECASE))
    for skill in SKILL_KEYWORDS
]


def extract_skills(raw_text: str, limit: int = 14):
    found = []
    for skill, pattern in _SKILL_PATTERNS:
        if pattern.search(raw_text):
            found.append(skill)
    return found[:limit]
