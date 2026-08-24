"""
CareerCast — Skill Gap Analysis Module (Milestone 3)

Given a resume's extracted skills and a target job category (one of the
top-5 predictions returned by the existing classifiers), this module:

1. Computes which required/recommended skills for that job are MISSING
   from the resume, and which are already PRESENT.
2. Generates concrete resume-wording suggestions: skills/keywords to ADD,
   and generic filler phrases to REMOVE.
3. Recommends courses to close the gap, via a pluggable search backend
   (defaults to YouTube Data API — free, instant API key, no partner
   approval needed, unlike Udemy/Coursera's official APIs).

This module is intentionally decoupled from the existing classifiers —
it takes plain Python data in (skills, job category) and returns plain
Python data out, so it can be called from either:
  - a new FastAPI endpoint (per Milestone 3's requirement), or
  - directly from the existing Flask app during a transition period.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import requests

# ---------------------------------------------------------------------------
# 1. SKILL TAXONOMY
# ---------------------------------------------------------------------------
# Maps a job category -> the skills expected for that role.
# "core"        = must-have skills; missing these is a significant gap.
# "recommended" = nice-to-have skills; missing these is a minor gap.
#
# NOTE: only Java Developer, Business Analyst, and Project Manager match
# your existing narrow classifier's categories exactly. Cloud Engineer is
# included here as an example since you mentioned it — you'll need to
# either (a) confirm Milestone 3 expands your classifier's job categories
# to include roles like this, or (b) keep this taxonomy independent of
# the classifier's categories (i.e. gap analysis runs on a broader skill
# ontology than what the classifier itself predicts). Add/edit freely —
# this dict is the single place to extend for new roles.
JOB_SKILLS: dict[str, dict[str, list[str]]] = {
    "Accountant": {
        "core": ["accounting", "bookkeeping", "tally", "gst", "financial statements", "excel"],
        "recommended": ["quickbooks", "sap fico", "taxation", "auditing", "reconciliation"],
    },
    "Advocate": {
        "core": ["legal research", "litigation", "contract drafting", "case law", "legal writing"],
        "recommended": ["negotiation", "compliance", "arbitration", "intellectual property"],
    },
    "Agriculture": {
        "core": ["crop management", "soil science", "irrigation", "agronomy", "pest control"],
        "recommended": ["gis", "precision farming", "supply chain", "sustainability"],
    },
    "Apparel": {
        "core": ["fashion design", "pattern making", "textile knowledge", "adobe illustrator", "merchandising"],
        "recommended": ["trend forecasting", "sourcing", "cad", "production planning"],
    },
    "Architecture": {
        "core": ["autocad", "revit", "building codes", "3d modeling", "site planning"],
        "recommended": ["sketchup", "sustainable design", "bim", "urban planning"],
    },
    "Arts": {
        "core": ["drawing", "adobe photoshop", "creativity", "portfolio development", "color theory"],
        "recommended": ["illustrator", "digital art", "sculpting", "art history"],
    },
    "Automobile": {
        "core": ["automotive systems", "cad", "vehicle diagnostics", "mechanical design", "quality testing"],
        "recommended": ["catia", "electric vehicles", "ansys", "manufacturing processes"],
    },
    "Aviation": {
        "core": ["flight operations", "aviation safety", "regulations", "aircraft systems", "logistics"],
        "recommended": ["faa/dgca compliance", "maintenance procedures", "air traffic coordination"],
    },
    "BPO": {
        "core": ["customer service", "communication", "crm software", "problem solving", "typing speed"],
        "recommended": ["upselling", "multitasking", "call handling", "sla management"],
    },
    "Banking": {
        "core": ["financial products", "kyc", "risk assessment", "customer service", "compliance"],
        "recommended": ["core banking systems", "loan processing", "aml", "credit analysis"],
    },
    "Blockchain": {
        "core": ["solidity", "smart contracts", "ethereum", "cryptography", "web3.js"],
        "recommended": ["hyperledger", "rust", "defi", "consensus algorithms", "ipfs"],
    },
    "Building and Construction": {
        "core": ["project scheduling", "autocad", "site management", "blueprint reading", "safety compliance"],
        "recommended": ["primavera", "cost estimation", "building codes", "quantity surveying"],
    },
    "Business Analyst": {
        "core": ["sql", "excel", "requirements gathering", "stakeholder management", "power bi"],
        "recommended": ["jira", "agile", "sql server", "tableau", "business process modeling"],
    },
    "Civil Engineer": {
        "core": ["autocad", "structural analysis", "surveying", "construction management", "staad pro"],
        "recommended": ["revit", "geotechnical engineering", "project scheduling", "quantity estimation"],
    },
    "Consultant": {
        "core": ["problem solving", "stakeholder management", "presentation skills", "data analysis", "client management"],
        "recommended": ["powerpoint", "strategy development", "change management", "market research"],
    },
    "Data Science": {
        "core": ["python", "pandas", "numpy", "scikit-learn", "sql", "statistics"],
        "recommended": ["tensorflow", "pytorch", "tableau", "spark", "docker", "mlflow"],
    },
    "Database": {
        "core": ["sql", "database design", "mysql", "indexing", "normalization"],
        "recommended": ["postgresql", "oracle", "mongodb", "performance tuning", "etl"],
    },
    "Designing": {
        "core": ["adobe photoshop", "adobe illustrator", "figma", "ui/ux", "typography"],
        "recommended": ["adobe xd", "wireframing", "branding", "prototyping"],
    },
    "DevOps": {
        "core": ["ci/cd", "docker", "kubernetes", "linux", "git", "aws"],
        "recommended": ["terraform", "jenkins", "ansible", "monitoring tools", "azure"],
    },
    "Digital Media": {
        "core": ["seo", "social media marketing", "content strategy", "google analytics", "adobe premiere"],
        "recommended": ["paid ads", "email marketing", "video editing", "canva"],
    },
    "DotNet Developer": {
        "core": [".net", "c#", "asp.net", "sql server", "mvc", "rest api"],
        "recommended": ["azure", "entity framework", "blazor", "microservices", "docker"],
    },
    "ETL Developer": {
        "core": ["etl", "sql", "informatica", "data warehousing", "ssis"],
        "recommended": ["python", "talend", "airflow", "snowflake", "data modeling"],
    },
    "Education": {
        "core": ["curriculum development", "classroom management", "lesson planning", "communication", "assessment design"],
        "recommended": ["e-learning tools", "learning management systems", "instructional design"],
    },
    "Electrical Engineering": {
        "core": ["circuit design", "matlab", "power systems", "pcb design", "autocad electrical"],
        "recommended": ["plc programming", "embedded systems", "control systems", "simulink"],
    },
    "Finance": {
        "core": ["financial modeling", "excel", "financial analysis", "valuation", "accounting"],
        "recommended": ["bloomberg terminal", "sap", "forecasting", "risk management"],
    },
    "Food and Beverages": {
        "core": ["food safety", "menu planning", "inventory management", "haccp", "customer service"],
        "recommended": ["cost control", "supply chain", "quality assurance", "nutrition knowledge"],
    },
    "Health and Fitness": {
        "core": ["exercise physiology", "nutrition", "fitness assessment", "program design", "client coaching"],
        "recommended": ["personal training certification", "group fitness", "wellness coaching"],
    },
    "Human Resources": {
        "core": ["recruitment", "onboarding", "hr policies", "performance management", "employee relations"],
        "recommended": ["hris systems", "payroll", "labor law", "talent management"],
    },
    "Information Technology": {
        "core": ["networking", "troubleshooting", "windows/linux administration", "sql", "it support"],
        "recommended": ["cloud platforms", "cybersecurity basics", "scripting", "itil"],
    },
    "Java Developer": {
        "core": ["java", "spring boot", "sql", "rest api", "git", "oop"],
        "recommended": ["microservices", "docker", "kubernetes", "junit", "kafka", "aws"],
    },
    "Management": {
        "core": ["leadership", "team management", "strategic planning", "communication", "budgeting"],
        "recommended": ["project management", "change management", "performance metrics", "negotiation"],
    },
    "Mechanical Engineer": {
        "core": ["autocad", "solidworks", "thermodynamics", "manufacturing processes", "product design"],
        "recommended": ["ansys", "cad/cam", "gd&t", "six sigma"],
    },
    "Network Security Engineer": {
        "core": ["network security", "firewalls", "vpn", "ids/ips", "linux"],
        "recommended": ["penetration testing", "siem tools", "ccna/ccnp", "cloud security"],
    },
    "Operations Manager": {
        "core": ["operations management", "process improvement", "supply chain", "budgeting", "team leadership"],
        "recommended": ["six sigma", "lean management", "inventory management", "kpi tracking"],
    },
    "PMO": {
        "core": ["project governance", "agile", "scrum", "risk management", "jira", "stakeholder management"],
        "recommended": ["pmp", "budgeting", "gantt charts", "confluence", "kanban", "resource planning"],
    },
    "Public Relations": {
        "core": ["media relations", "press releases", "communication", "crisis management", "content writing"],
        "recommended": ["social media strategy", "brand management", "event planning"],
    },
    "Python Developer": {
        "core": ["python", "django", "flask", "sql", "rest api", "git"],
        "recommended": ["docker", "aws", "celery", "postgresql", "unit testing"],
    },
    "React Developer": {
        "core": ["react", "javascript", "html", "css", "redux", "rest api"],
        "recommended": ["typescript", "next.js", "node.js", "jest", "webpack"],
    },
    "SAP Developer": {
        "core": ["sap abap", "sap modules", "sql", "sap fiori", "business processes"],
        "recommended": ["sap hana", "sap ui5", "sap mm/sd/fi", "odata"],
    },
    "SQL Developer": {
        "core": ["sql", "stored procedures", "database design", "query optimization", "t-sql"],
        "recommended": ["ssrs", "ssis", "postgresql", "performance tuning"],
    },
    "Sales": {
        "core": ["negotiation", "crm software", "lead generation", "communication", "cold calling"],
        "recommended": ["salesforce", "account management", "b2b sales", "pipeline management"],
    },
    "Testing": {
        "core": ["manual testing", "test case design", "selenium", "bug tracking", "sql"],
        "recommended": ["automation testing", "api testing", "jmeter", "agile testing", "ci/cd"],
    },
    "Web Designing": {
        "core": ["html", "css", "javascript", "responsive design", "figma"],
        "recommended": ["wordpress", "adobe xd", "bootstrap", "accessibility", "seo basics"],
    },
}

# Generic resume filler phrases worth flagging for removal — these add
# noise/word count without signaling skill, and are commonly flagged by
# ATS-optimization guides.
FILLER_PHRASES = [
    "hardworking", "team player", "detail oriented", "go-getter",
    "results driven", "self motivated", "dynamic individual",
    "passionate about", "think outside the box", "synergy",
]


@dataclass
class GapReport:
    job_category: str
    present_core: list[str] = field(default_factory=list)
    missing_core: list[str] = field(default_factory=list)
    present_recommended: list[str] = field(default_factory=list)
    missing_recommended: list[str] = field(default_factory=list)
    filler_phrases_found: list[str] = field(default_factory=list)
    add_suggestions: list[str] = field(default_factory=list)
    remove_suggestions: list[str] = field(default_factory=list)
    course_suggestions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_category": self.job_category,
            "skills": {
                "present_core": self.present_core,
                "missing_core": self.missing_core,
                "present_recommended": self.present_recommended,
                "missing_recommended": self.missing_recommended,
            },
            "resume_suggestions": {
                "add": self.add_suggestions,
                "remove": self.remove_suggestions,
            },
            "courses": self.course_suggestions,
        }


# ---------------------------------------------------------------------------
# 2. GAP COMPUTATION
# ---------------------------------------------------------------------------
def _normalize(skill: str) -> str:
    return re.sub(r"\s+", " ", skill.strip().lower())


def compute_gap(resume_skills: list[str], job_category: str) -> GapReport:
    """
    resume_skills: list of skills already extracted from the resume
                   (reuse your existing skill-extraction logic — this
                   function assumes that list is already produced).
    job_category: one of the keys in JOB_SKILLS (e.g. "Cloud Engineer").
    """
    if job_category not in JOB_SKILLS:
        raise ValueError(
            f"Unknown job category '{job_category}'. "
            f"Add it to JOB_SKILLS first. Known: {list(JOB_SKILLS)}"
        )

    normalized_resume = {_normalize(s) for s in resume_skills}
    taxonomy = JOB_SKILLS[job_category]

    report = GapReport(job_category=job_category)

    for skill in taxonomy["core"]:
        (report.present_core if _normalize(skill) in normalized_resume
         else report.missing_core).append(skill)

    for skill in taxonomy["recommended"]:
        (report.present_recommended if _normalize(skill) in normalized_resume
         else report.missing_recommended).append(skill)

    return report


# ---------------------------------------------------------------------------
# 3. RESUME WORDING SUGGESTIONS
# ---------------------------------------------------------------------------
def generate_wording_suggestions(resume_text: str, report: GapReport) -> None:
    """Mutates `report` in place, filling add_suggestions / remove_suggestions."""

    # ADD: missing core skills first (highest impact), then recommended
    for skill in report.missing_core:
        report.add_suggestions.append(
            f"Add a bullet demonstrating hands-on experience with '{skill}' — "
            f"this is a core requirement for {report.job_category} and is "
            f"currently absent from your resume."
        )
    for skill in report.missing_recommended:
        report.add_suggestions.append(
            f"Consider mentioning '{skill}' if you have any exposure to it — "
            f"it's commonly expected for {report.job_category} roles."
        )

    # REMOVE: generic filler phrases found in the resume text
    lowered = resume_text.lower()
    for phrase in FILLER_PHRASES:
        if phrase in lowered:
            report.filler_phrases_found.append(phrase)
            report.remove_suggestions.append(
                f"Remove or replace generic phrase '{phrase}' — it doesn't "
                f"signal a concrete skill and wastes limited resume space. "
                f"Replace with a specific, measurable achievement instead."
            )


# ---------------------------------------------------------------------------
# 4. COURSE RECOMMENDATIONS (pluggable search backend)
# ---------------------------------------------------------------------------
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


def search_courses_youtube(query: str, max_results: int = 3) -> list[dict]:
    """
    Searches YouTube for course/tutorial content matching `query`.
    Requires YOUTUBE_API_KEY env var — get a free key instantly at
    https://console.cloud.google.com/apis/credentials (enable "YouTube Data API v3").
    """
    if not YOUTUBE_API_KEY:
        return [{
            "error": "YOUTUBE_API_KEY not set. Get a free key at "
                     "console.cloud.google.com/apis/credentials and set it "
                     "as an environment variable."
        }]

    params = {
        "part": "snippet",
        "q": f"{query} full course tutorial",
        "type": "video",
        "maxResults": max_results,
        "order": "relevance",
        "key": YOUTUBE_API_KEY,
    }
    try:
        resp = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        return [{"error": f"Course search failed: {exc}"}]

    results = []
    for item in data.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})
        if not video_id:
            continue
        results.append({
            "title": snippet.get("title"),
            "channel": snippet.get("channelTitle"),
            "url": f"https://www.youtube.com/watch?v={video_id}",
        })
    return results


def add_course_suggestions(report: GapReport, search_fn=search_courses_youtube) -> None:
    """
    Mutates `report` in place. Searches for courses covering each missing
    CORE skill first (highest priority gaps). Limited to top 3 missing
    core skills to keep API usage/latency reasonable per request.
    `search_fn` is swappable — pass a different function here (e.g. a
    Udemy/Coursera search) once you have API access to one of those.
    """
    for skill in report.missing_core[:3]:
        courses = search_fn(skill)
        report.course_suggestions.append({
            "skill": skill,
            "results": courses,
        })


# ---------------------------------------------------------------------------
# 5. TOP-LEVEL ENTRY POINT
# ---------------------------------------------------------------------------
def generate_gap_report(resume_text: str, resume_skills: list[str], job_category: str) -> dict:
    """
    The single function your FastAPI endpoint should call.

    resume_text:   raw/cleaned resume text (for filler-phrase detection)
    resume_skills: skills already extracted by your existing extraction logic
    job_category:  the job the user clicked on, from their top-5 predictions
    """
    report = compute_gap(resume_skills, job_category)
    generate_wording_suggestions(resume_text, report)
    add_course_suggestions(report)
    return report.to_dict()


if __name__ == "__main__":
    # Quick manual test — run: python skill_gap.py
    sample_resume_text = """
    I am a hardworking and results driven engineer with experience in
    Python, Docker, and Linux. Passionate about building scalable systems.
    """
    sample_skills = ["python", "docker", "linux"]

    result = generate_gap_report(sample_resume_text, sample_skills, "Cloud Engineer")
    import json
    print(json.dumps(result, indent=2))