# CareerCast — AI-Powered Career Intelligence Platform

CareerCast matches a resume against a set of job categories using a trained
Logistic Regression classifier. Paste resume text or upload a **PDF / DOCX /
TXT** file, and get back the top matching job categories with confidence
scores, along with identified skills and role-specific skill suggestions.

**Live demo:**https://ai-powered-career-intelligence-plat-eta.vercel.app/

---

## Screenshots

**Home screen**

![Home screen](screenshots/01-home.png)

**Pasted resume text — ranked job matches, identified skills, and suggested skills**

![Paste text result](screenshots/02-paste-text-result.png)

**File upload — DOCX resume analyzed with skills breakdown**

![Upload file result](screenshots/03-upload-file-result.png)

---

## Features

- Paste resume text directly, or upload a **PDF, DOCX, or TXT** file
- Text extraction via `pdfplumber` (PDF) and `python-docx` (DOCX)
- Same text-cleaning pipeline at inference time as at training time
  (lowercasing, URL/email/phone stripping, punctuation removal, stopword
  removal, lemmatization) so the TF-IDF vectorizer sees consistent input
- Top 5 job category matches with confidence percentages
  (`model.predict_proba()`)
- Identified skills pulled from the resume, plus suggested skills for the
  top-matched role
- Animated circular gauge for the #1 match and ranked signal bars for the rest

## Tech stack

- **Backend:** Flask, scikit-learn (Logistic Regression + TF-IDF), joblib
- **File parsing:** pdfplumber, python-docx
- **Frontend:** single-page HTML/CSS/JS template (no build step)

## Dataset

The classifier is trained on the [Resume Classification Dataset](https://media.githubusercontent.com/media/noran-mohamed/Resume-Classification-Dataset/refs/heads/main/Dataset.csv)
by [noran-mohamed](https://github.com/noran-mohamed/Resume-Classification-Dataset),
a labeled collection of resumes across multiple job categories used to train
the TF-IDF + Logistic Regression pipeline in `train_model.py`.

## Project structure

```
.
├── app.py                          # Flask server: file parsing, cleaning, prediction
├── train_model.py                  # Training pipeline -> resume_job_classifier.joblib
├── templates/
│   └── index.html                  # UI
├── screenshots/                    # Images used in this README
├── requirements.txt
└── README.md
```

## Getting started (run locally)

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/AI-powered-career-intelligence-platform.git
cd AI-powered-career-intelligence-platform
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get the trained model file

If you already ran the notebook/script and it produced
`resume_job_classifier.joblib`, copy that file into the project root, next
to `app.py`.

Otherwise, train it yourself (downloads the dataset, needs internet access):

```bash
python train_model.py
```

This runs a grid search over a few hyperparameter combinations and takes a
few minutes. It ends with `resume_job_classifier.joblib` saved in the
current folder.

### 5. Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Deploying to Vercel

Vercel runs Python as serverless functions rather than a long-lived Flask
process, so a couple of adjustments are needed on top of what works locally:

1. **Add a `vercel.json`** at the project root:

   ```json
   {
     "builds": [
       { "src": "app.py", "use": "@vercel/python" }
     ],
     "routes": [
       { "src": "/(.*)", "dest": "app.py" }
     ]
   }
   ```

2. **Keep `requirements.txt` at the project root** — Vercel installs from it
   automatically for the Python build.

3. **Model file size matters.** Vercel serverless functions have a deployed
   size limit (250 MB uncompressed, including dependencies like
   scikit-learn/pandas). If `resume_job_classifier.joblib` plus your
   dependencies push past that, either trim dependencies, or host the model
   file externally (e.g. a small object store) and load it at cold start,
   or use a platform built for long-running Python apps instead
   (Render, Railway, Fly.io, or a VM) — these avoid the serverless size/time
   limits entirely and are usually simpler for scikit-learn apps.

4. **Deploy:**

   ```bash
   npm install -g vercel
   vercel login
   vercel --prod
   ```

   Or connect the GitHub repo directly in the Vercel dashboard
   (New Project → Import Git Repository) so every push to `main` auto-deploys.

5. Once deployed, update the **Live demo** link at the top of this README.

---

## Notes

- Max upload size is 8MB.
- If `resume_job_classifier.joblib` isn't found next to `app.py`, the app
  still loads but shows a banner explaining the model isn't ready, and
  disables the "Analyze resume" button.
- The percentages are the model's own confidence estimates
  (`predict_proba`), not a certified skills match — they reflect how
  similar the resume's language is to resumes in each training category.
