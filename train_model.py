"""
Train the Resume -> Suitable Job classifier used by app.py.
=============================================================
This is your existing Logistic Regression pipeline (from the Infosys
Springboard notebook), lightly reorganized so it can be run as a standalone
script and saves the artifact CareerCast's web app expects.

Run this in Colab, Jupyter, or locally — anywhere with internet access:
    pip install pandas numpy scikit-learn nltk joblib
    python train_model.py

Output:
    resume_job_classifier.joblib   <-- copy this next to app.py
"""

import re
import string

import joblib
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    top_k_accuracy_score,
)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

DATA_URL = (
    "https://media.githubusercontent.com/media/noran-mohamed/"
    "Resume-Classification-Dataset/refs/heads/main/Dataset.csv"
)

RANDOM_STATE = 42
TOP_K = 5
MODEL_OUT = "resume_job_classifier.joblib"


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    expected = {"Category", "Text"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(
            f"Downloaded file doesn't have the expected columns.\n"
            f"Expected: {expected}\nGot columns: {list(df.columns)}"
        )
    df = df.dropna(subset=["Category", "Text"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Text"]).reset_index(drop=True)
    return df


URL_RE = re.compile(r"http\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


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


def build_and_tune(X_train, y_train) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.9)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    param_grid = {
        "tfidf__max_features": [15000, 25000, 40000],
        "clf__C": [1, 5, 10, 20],
        "clf__solver": ["liblinear", "lbfgs"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    return search.best_estimator_


def evaluate(model, X_test, y_test, classes, k=TOP_K):
    y_pred = model.predict(X_test)
    top1_acc = accuracy_score(y_test, y_pred)
    y_proba = model.predict_proba(X_test)
    topk_acc = top_k_accuracy_score(y_test, y_proba, k=k, labels=classes)
    print(f"\nTop-1 accuracy: {top1_acc:.4f}")
    print(f"Top-{k} accuracy: {topk_acc:.4f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    return top1_acc, topk_acc


def main():
    print("Loading dataset...")
    df = load_data()
    print(f"Loaded {len(df)} resumes across {df['Category'].nunique()} categories.")

    print("Cleaning text...")
    df["clean_text"] = df["Text"].astype(str).apply(clean_resume_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    counts = df["Category"].value_counts()
    valid_categories = counts[counts >= 5].index
    df = df[df["Category"].isin(valid_categories)].reset_index(drop=True)

    X = df["clean_text"]
    y = df["Category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Training + tuning Logistic Regression...")
    model = build_and_tune(X_train, y_train)

    classes = sorted(y.unique())
    evaluate(model, X_test, y_test, classes, k=TOP_K)

    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")
    print("Copy this file next to app.py, then run: python app.py")


if __name__ == "__main__":
    main()
