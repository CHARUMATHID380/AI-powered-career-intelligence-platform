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
# ---------------------------------------------------------------------------
# 4. COURSE CATALOGUE  (Udemy · Coursera · Global Certifications)
# ---------------------------------------------------------------------------
# Curated real courses — no API key required.  Each entry has:
#   title    : course / certification name
#   platform : Udemy | Coursera | Certification
#   url      : direct link to the course / credential
#   provider : who offers it (university, vendor, org)
#
# The fallback key "DEFAULT" is used when a skill has no specific entry.
# ---------------------------------------------------------------------------
COURSE_CATALOGUE: dict[str, list[dict]] = {
    # ── Programming languages ──────────────────────────────────────────────
    "python": [
        {"title": "2023 Complete Python Bootcamp", "platform": "Udemy",
         "url": "https://www.udemy.com/course/complete-python-bootcamp/",
         "provider": "Jose Portilla"},
        {"title": "Python for Everybody Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/python",
         "provider": "University of Michigan"},
        {"title": "PCEP – Certified Entry-Level Python Programmer", "platform": "Certification",
         "url": "https://pythoninstitute.org/pcep",
         "provider": "Python Institute"},
    ],
    "java": [
        {"title": "Java Programming Masterclass", "platform": "Udemy",
         "url": "https://www.udemy.com/course/java-the-complete-java-developer-course/",
         "provider": "Tim Buchalka"},
        {"title": "Object-Oriented Programming in Java", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/object-oriented-programming",
         "provider": "Duke University"},
        {"title": "Oracle Certified Professional – Java SE", "platform": "Certification",
         "url": "https://education.oracle.com/java-se-17-developer/pexam_1Z0-829",
         "provider": "Oracle"},
    ],
    "sql": [
        {"title": "The Complete SQL Bootcamp", "platform": "Udemy",
         "url": "https://www.udemy.com/course/the-complete-sql-bootcamp/",
         "provider": "Jose Portilla"},
        {"title": "SQL for Data Science", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/sql-for-data-science",
         "provider": "UC Davis"},
        {"title": "Microsoft Certified: Azure Data Fundamentals (DP-900)", "platform": "Certification",
         "url": "https://learn.microsoft.com/en-us/credentials/certifications/azure-data-fundamentals/",
         "provider": "Microsoft"},
    ],
    "javascript": [
        {"title": "The Complete JavaScript Course", "platform": "Udemy",
         "url": "https://www.udemy.com/course/the-complete-javascript-course/",
         "provider": "Jonas Schmedtmann"},
        {"title": "JavaScript Algorithms and Data Structures", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/algorithms-part1",
         "provider": "Princeton University"},
    ],
    "r": [
        {"title": "R Programming A-Z", "platform": "Udemy",
         "url": "https://www.udemy.com/course/r-programming/",
         "provider": "SuperDataScience"},
        {"title": "Data Science: Statistics and Machine Learning Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/data-science-statistics-machine-learning",
         "provider": "Johns Hopkins University"},
    ],
    "c++": [
        {"title": "Beginning C++ Programming", "platform": "Udemy",
         "url": "https://www.udemy.com/course/beginning-c-plus-plus-programming/",
         "provider": "Tim Buchalka"},
        {"title": "C++ For C Programmers", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/c-plus-plus-a",
         "provider": "UC Santa Cruz"},
    ],
    # ── ML / Data Science ──────────────────────────────────────────────────
    "machine learning": [
        {"title": "Machine Learning A-Z", "platform": "Udemy",
         "url": "https://www.udemy.com/course/machinelearning/",
         "provider": "Kirill Eremenko"},
        {"title": "Machine Learning Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/machine-learning-introduction",
         "provider": "Andrew Ng / Stanford"},
        {"title": "TensorFlow Developer Certificate", "platform": "Certification",
         "url": "https://www.tensorflow.org/certificate",
         "provider": "Google"},
    ],
    "deep learning": [
        {"title": "Deep Learning A-Z 2024", "platform": "Udemy",
         "url": "https://www.udemy.com/course/deeplearning/",
         "provider": "Kirill Eremenko"},
        {"title": "Deep Learning Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/deep-learning",
         "provider": "Andrew Ng / deeplearning.ai"},
    ],
    "data science": [
        {"title": "Python for Data Science and Machine Learning Bootcamp", "platform": "Udemy",
         "url": "https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/",
         "provider": "Jose Portilla"},
        {"title": "IBM Data Science Professional Certificate", "platform": "Coursera",
         "url": "https://www.coursera.org/professional-certificates/ibm-data-science",
         "provider": "IBM"},
        {"title": "Certified Analytics Professional (CAP)", "platform": "Certification",
         "url": "https://www.certifiedanalytics.org/",
         "provider": "INFORMS"},
    ],
    "data analysis": [
        {"title": "Data Analysis with Pandas and Python", "platform": "Udemy",
         "url": "https://www.udemy.com/course/data-analysis-with-pandas/",
         "provider": "Boris Paskhaver"},
        {"title": "Google Data Analytics Professional Certificate", "platform": "Coursera",
         "url": "https://www.coursera.org/professional-certificates/google-data-analytics",
         "provider": "Google"},
    ],
    "statistics": [
        {"title": "Statistics & Probability in Data Science using Python", "platform": "Udemy",
         "url": "https://www.udemy.com/course/statistics-probability/",
         "provider": "Jose Portilla"},
        {"title": "Statistics with Python Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/statistics-with-python",
         "provider": "University of Michigan"},
    ],
    "tensorflow": [
        {"title": "TensorFlow Developer Certificate Bootcamp", "platform": "Udemy",
         "url": "https://www.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery/",
         "provider": "Zero to Mastery"},
        {"title": "TensorFlow: Advanced Techniques Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/tensorflow-advanced-techniques",
         "provider": "deeplearning.ai"},
        {"title": "TensorFlow Developer Certificate", "platform": "Certification",
         "url": "https://www.tensorflow.org/certificate",
         "provider": "Google"},
    ],
    "pytorch": [
        {"title": "PyTorch for Deep Learning Bootcamp", "platform": "Udemy",
         "url": "https://www.udemy.com/course/pytorch-for-deep-learning-and-computer-vision/",
         "provider": "Zero to Mastery"},
        {"title": "Deep Neural Networks with PyTorch", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/deep-neural-networks-with-pytorch",
         "provider": "IBM"},
    ],
    "natural language processing": [
        {"title": "NLP – Natural Language Processing with Python", "platform": "Udemy",
         "url": "https://www.udemy.com/course/nlp-natural-language-processing-with-python/",
         "provider": "Jose Portilla"},
        {"title": "Natural Language Processing Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/natural-language-processing",
         "provider": "deeplearning.ai"},
    ],
    # ── Cloud / DevOps ─────────────────────────────────────────────────────
    "aws": [
        {"title": "AWS Certified Solutions Architect – Ultimate Exam Training", "platform": "Udemy",
         "url": "https://www.udemy.com/course/aws-certified-solutions-architect-associate-saa-c03/",
         "provider": "Stéphane Maarek"},
        {"title": "AWS Cloud Technology & Services", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/aws-cloud-technical-essentials",
         "provider": "Amazon Web Services"},
        {"title": "AWS Certified Solutions Architect – Associate (SAA-C03)", "platform": "Certification",
         "url": "https://aws.amazon.com/certification/certified-solutions-architect-associate/",
         "provider": "Amazon Web Services"},
    ],
    "azure": [
        {"title": "AZ-900 Microsoft Azure Fundamentals Exam Prep", "platform": "Udemy",
         "url": "https://www.udemy.com/course/az900-azure/",
         "provider": "Scott Duffy"},
        {"title": "Microsoft Azure Fundamentals (AZ-900)", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/microsoft-azure-fundamentals-az-900",
         "provider": "Microsoft"},
        {"title": "Microsoft Certified: Azure Administrator Associate (AZ-104)", "platform": "Certification",
         "url": "https://learn.microsoft.com/en-us/credentials/certifications/azure-administrator/",
         "provider": "Microsoft"},
    ],
    "google cloud": [
        {"title": "Google Cloud Professional Data Engineer", "platform": "Udemy",
         "url": "https://www.udemy.com/course/gcp-data-engineer-and-cloud-architect/",
         "provider": "QwikLabs"},
        {"title": "Google Cloud Fundamentals: Core Infrastructure", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/gcp-fundamentals",
         "provider": "Google Cloud"},
        {"title": "Google Cloud Professional Cloud Architect", "platform": "Certification",
         "url": "https://cloud.google.com/learn/certification/cloud-architect",
         "provider": "Google Cloud"},
    ],
    "docker": [
        {"title": "Docker & Kubernetes: The Complete Guide", "platform": "Udemy",
         "url": "https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/",
         "provider": "Stephen Grider"},
        {"title": "Docker for the Absolute Beginner – Hands-On", "platform": "Udemy",
         "url": "https://www.udemy.com/course/learn-docker/",
         "provider": "KodeKloud"},
    ],
    "kubernetes": [
        {"title": "Kubernetes for the Absolute Beginners", "platform": "Udemy",
         "url": "https://www.udemy.com/course/learn-kubernetes/",
         "provider": "KodeKloud"},
        {"title": "Certified Kubernetes Administrator (CKA)", "platform": "Certification",
         "url": "https://training.linuxfoundation.org/certification/certified-kubernetes-administrator-cka/",
         "provider": "Linux Foundation"},
    ],
    "devops": [
        {"title": "DevOps Beginners to Advanced with Projects", "platform": "Udemy",
         "url": "https://www.udemy.com/course/devsecops/",
         "provider": "Imran Teli"},
        {"title": "DevOps on AWS Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/aws-devops",
         "provider": "Amazon Web Services"},
        {"title": "AWS Certified DevOps Engineer – Professional", "platform": "Certification",
         "url": "https://aws.amazon.com/certification/certified-devops-engineer-professional/",
         "provider": "Amazon Web Services"},
    ],
    "ci/cd": [
        {"title": "Jenkins, From Zero To Hero: Become a DevOps Jenkins Master", "platform": "Udemy",
         "url": "https://www.udemy.com/course/jenkins-from-zero-to-hero/",
         "provider": "Ricardo Andre Gonzalez"},
        {"title": "Continuous Integration and Continuous Delivery (CI/CD)", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/continuous-integration-and-delivery-ci-cd",
         "provider": "IBM"},
    ],
    "linux": [
        {"title": "Linux Command Line Basics", "platform": "Udemy",
         "url": "https://www.udemy.com/course/linux-command-line-volume1/",
         "provider": "Jason Cannon"},
        {"title": "Linux and Bash for Data Engineering", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/linux-and-bash-for-data-engineering-duke",
         "provider": "Duke University"},
        {"title": "LPIC-1: Linux Administrator", "platform": "Certification",
         "url": "https://www.lpi.org/our-certifications/lpic-1-overview",
         "provider": "Linux Professional Institute"},
    ],
    # ── Databases / BI ─────────────────────────────────────────────────────
    "mysql": [
        {"title": "The Ultimate MySQL Bootcamp", "platform": "Udemy",
         "url": "https://www.udemy.com/course/the-ultimate-mysql-bootcamp-go-from-sql-beginner-to-expert/",
         "provider": "Colt Steele"},
        {"title": "MySQL for Data Analytics and Business Intelligence", "platform": "Udemy",
         "url": "https://www.udemy.com/course/mysql-data-analytics-business-intelligence/",
         "provider": "365 Data Science"},
    ],
    "postgresql": [
        {"title": "SQL and PostgreSQL: The Complete Developer's Guide", "platform": "Udemy",
         "url": "https://www.udemy.com/course/sql-and-postgresql/",
         "provider": "Stephen Grider"},
    ],
    "mongodb": [
        {"title": "MongoDB – The Complete Developer's Guide", "platform": "Udemy",
         "url": "https://www.udemy.com/course/mongodb-the-complete-developers-guide/",
         "provider": "Maximilian Schwarzmüller"},
        {"title": "MongoDB Node.js Developer Path", "platform": "Certification",
         "url": "https://learn.mongodb.com/learning-paths/mongodb-nodejs-developer-path",
         "provider": "MongoDB University"},
    ],
    "power bi": [
        {"title": "Microsoft Power BI Desktop for Business Intelligence", "platform": "Udemy",
         "url": "https://www.udemy.com/course/microsoft-power-bi-up-running-with-power-bi-desktop/",
         "provider": "Maven Analytics"},
        {"title": "Microsoft Power BI Data Analyst Associate (PL-300)", "platform": "Certification",
         "url": "https://learn.microsoft.com/en-us/credentials/certifications/power-bi-data-analyst-associate/",
         "provider": "Microsoft"},
    ],
    "tableau": [
        {"title": "Tableau 2024 A-Z: Hands-On Tableau Training", "platform": "Udemy",
         "url": "https://www.udemy.com/course/tableau10/",
         "provider": "Kirill Eremenko"},
        {"title": "Data Visualization with Tableau Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/data-visualization",
         "provider": "UC Davis"},
        {"title": "Tableau Desktop Specialist", "platform": "Certification",
         "url": "https://www.tableau.com/learn/certification/desktop-specialist",
         "provider": "Salesforce Tableau"},
    ],
    "excel": [
        {"title": "Microsoft Excel – Excel from Beginner to Advanced", "platform": "Udemy",
         "url": "https://www.udemy.com/course/microsoft-excel-2013-from-beginner-to-advanced-and-beyond/",
         "provider": "Kyle Pew"},
        {"title": "Excel Skills for Business Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/excel",
         "provider": "Macquarie University"},
        {"title": "Microsoft Office Specialist: Excel Expert", "platform": "Certification",
         "url": "https://learn.microsoft.com/en-us/credentials/certifications/mos-excel-expert-2019/",
         "provider": "Microsoft"},
    ],
    # ── Web / Frontend ─────────────────────────────────────────────────────
    "react": [
        {"title": "React – The Complete Guide (incl. React Router & Redux)", "platform": "Udemy",
         "url": "https://www.udemy.com/course/react-the-complete-guide-incl-redux/",
         "provider": "Maximilian Schwarzmüller"},
        {"title": "Front-End Web Development with React", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/front-end-react",
         "provider": "The Hong Kong University of Science and Technology"},
    ],
    "node.js": [
        {"title": "The Complete Node.js Developer Course", "platform": "Udemy",
         "url": "https://www.udemy.com/course/the-complete-nodejs-developer-course-2/",
         "provider": "Andrew Mead"},
        {"title": "Server-side Development with NodeJS, Express and MongoDB", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/server-side-nodejs",
         "provider": "The Hong Kong University of Science and Technology"},
    ],
    "rest api": [
        {"title": "REST APIs with Flask and Python", "platform": "Udemy",
         "url": "https://www.udemy.com/course/rest-api-flask-and-python/",
         "provider": "Jose Salvatierra"},
        {"title": "API Design and Fundamentals of Google Cloud's Apigee API Platform", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/api-design-apigee-gcp",
         "provider": "Google Cloud"},
    ],
    # ── Project Management ──────────────────────────────────────────────────
    "project management": [
        {"title": "PMP Exam Prep Seminar – PMBOK 7th Edition", "platform": "Udemy",
         "url": "https://www.udemy.com/course/pmp-pmbok6-exam-prep/",
         "provider": "Joseph Phillips"},
        {"title": "Google Project Management: Professional Certificate", "platform": "Coursera",
         "url": "https://www.coursera.org/professional-certificates/google-project-management",
         "provider": "Google"},
        {"title": "Project Management Professional (PMP)", "platform": "Certification",
         "url": "https://www.pmi.org/certifications/project-management-pmp",
         "provider": "PMI"},
    ],
    "agile": [
        {"title": "Agile Fundamentals: Including Scrum & Kanban", "platform": "Udemy",
         "url": "https://www.udemy.com/course/agile-fundamentals-including-scrum-and-kanban-2019/",
         "provider": "Value Insights"},
        {"title": "Agile Development Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/agile-development",
         "provider": "University of Virginia"},
        {"title": "PMI-ACP (Agile Certified Practitioner)", "platform": "Certification",
         "url": "https://www.pmi.org/certifications/agile-acp",
         "provider": "PMI"},
    ],
    "scrum": [
        {"title": "Scrum Master Certification Preparation", "platform": "Udemy",
         "url": "https://www.udemy.com/course/scrum-master-certification-preparation-mock-exam-questions-psm-i/",
         "provider": "Frank Turley"},
        {"title": "Professional Scrum Master (PSM I)", "platform": "Certification",
         "url": "https://www.scrum.org/assessments/professional-scrum-master-i-certification",
         "provider": "Scrum.org"},
        {"title": "Certified ScrumMaster (CSM)", "platform": "Certification",
         "url": "https://www.scrumalliance.org/get-certified/scrum-master-track/certified-scrummaster",
         "provider": "Scrum Alliance"},
    ],
    # ── Finance / Accounting ────────────────────────────────────────────────
    "accounting": [
        {"title": "Accounting Fundamentals", "platform": "Udemy",
         "url": "https://www.udemy.com/course/accounting-fundamentals/",
         "provider": "Robert (Bob) Steele"},
        {"title": "Introduction to Financial Accounting", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/wharton-accounting",
         "provider": "University of Pennsylvania (Wharton)"},
        {"title": "CPA – Certified Public Accountant", "platform": "Certification",
         "url": "https://www.aicpa-cima.com/certifications/cpa",
         "provider": "AICPA"},
    ],
    "financial modeling": [
        {"title": "Financial Modeling & Valuation Analyst (FMVA) Prep", "platform": "Udemy",
         "url": "https://www.udemy.com/course/the-complete-financial-analyst-course/",
         "provider": "365 Careers"},
        {"title": "Financial Modeling for Startups & Small Businesses", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/financial-modeling",
         "provider": "Coursera Project Network"},
        {"title": "Financial Modeling & Valuation Analyst (FMVA)", "platform": "Certification",
         "url": "https://corporatefinanceinstitute.com/certifications/financial-modeling-valuation-analyst-fmva-program/",
         "provider": "Corporate Finance Institute (CFI)"},
    ],
    "finance": [
        {"title": "Complete Financial Analyst Training & Investing Course", "platform": "Udemy",
         "url": "https://www.udemy.com/course/the-complete-financial-analyst-course/",
         "provider": "365 Careers"},
        {"title": "Finance for Everyone Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/finance-for-everyone",
         "provider": "McMaster University"},
        {"title": "CFA (Chartered Financial Analyst)", "platform": "Certification",
         "url": "https://www.cfainstitute.org/programs/cfa",
         "provider": "CFA Institute"},
    ],
    "auditing": [
        {"title": "Accounting: From Beginner to Advanced", "platform": "Udemy",
         "url": "https://www.udemy.com/course/accounting-from-beginner-to-advanced/",
         "provider": "Robert (Bob) Steele"},
        {"title": "CIA – Certified Internal Auditor", "platform": "Certification",
         "url": "https://www.theiia.org/en/certifications/cia/",
         "provider": "The Institute of Internal Auditors"},
    ],
    # ── Business / HR / Marketing ───────────────────────────────────────────
    "business analysis": [
        {"title": "Business Analysis Fundamentals", "platform": "Udemy",
         "url": "https://www.udemy.com/course/business-analysis-ba/",
         "provider": "Jeremy Aschenbrenner"},
        {"title": "Business Analysis & Process Management", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/business-analysis-process-management",
         "provider": "Coursera Project Network"},
        {"title": "CBAP – Certified Business Analysis Professional", "platform": "Certification",
         "url": "https://www.iiba.org/professional-development/certifications/cbap/",
         "provider": "IIBA"},
    ],
    "marketing": [
        {"title": "The Complete Digital Marketing Course", "platform": "Udemy",
         "url": "https://www.udemy.com/course/learn-digital-marketing-course/",
         "provider": "Rob Percival"},
        {"title": "Google Digital Marketing & E-commerce Certificate", "platform": "Coursera",
         "url": "https://www.coursera.org/professional-certificates/google-digital-marketing-ecommerce",
         "provider": "Google"},
    ],
    "seo": [
        {"title": "SEO 2024: Complete SEO Training + SEO for WordPress", "platform": "Udemy",
         "url": "https://www.udemy.com/course/seo-training/",
         "provider": "Brad Hussey"},
        {"title": "Google SEO Fundamentals", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/seo-fundamentals",
         "provider": "UC Davis"},
    ],
    "leadership": [
        {"title": "Leadership: Practical Leadership Skills", "platform": "Udemy",
         "url": "https://www.udemy.com/course/leadership-practical-leadership-skills/",
         "provider": "Expert Academy"},
        {"title": "Leading People and Teams Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/leading-teams",
         "provider": "University of Michigan"},
    ],
    # ── Cyber Security ──────────────────────────────────────────────────────
    "network security": [
        {"title": "The Complete Cyber Security Course: Network Security!", "platform": "Udemy",
         "url": "https://www.udemy.com/course/network-security-course/",
         "provider": "Nathan House"},
        {"title": "Google Cybersecurity Professional Certificate", "platform": "Coursera",
         "url": "https://www.coursera.org/professional-certificates/google-cybersecurity",
         "provider": "Google"},
        {"title": "CompTIA Security+", "platform": "Certification",
         "url": "https://www.comptia.org/certifications/security",
         "provider": "CompTIA"},
    ],
    "penetration testing": [
        {"title": "Learn Ethical Hacking From Scratch", "platform": "Udemy",
         "url": "https://www.udemy.com/course/learn-ethical-hacking-from-scratch/",
         "provider": "Zaid Sabih"},
        {"title": "CEH – Certified Ethical Hacker", "platform": "Certification",
         "url": "https://www.eccouncil.org/programs/certified-ethical-hacker-ceh/",
         "provider": "EC-Council"},
        {"title": "OSCP – Offensive Security Certified Professional", "platform": "Certification",
         "url": "https://www.offsec.com/courses/pen-200/",
         "provider": "Offensive Security"},
    ],
    # ── Design / UI/UX ──────────────────────────────────────────────────────
    "ui/ux": [
        {"title": "User Experience Design Essentials – Adobe XD", "platform": "Udemy",
         "url": "https://www.udemy.com/course/ui-ux-web-design-using-adobe-xd/",
         "provider": "Daniel Walter Scott"},
        {"title": "Google UX Design Professional Certificate", "platform": "Coursera",
         "url": "https://www.coursera.org/professional-certificates/google-ux-design",
         "provider": "Google"},
    ],
    # ── Spring Boot / Java Backend ──────────────────────────────────────────
    "spring boot": [
        {"title": "Spring Boot 3, Spring 6 & Hibernate for Beginners", "platform": "Udemy",
         "url": "https://www.udemy.com/course/spring-hibernate-tutorial/",
         "provider": "Chad Darby"},
        {"title": "Building Scalable Java Microservices with Spring Boot and Spring Cloud", "platform": "Coursera",
         "url": "https://www.coursera.org/learn/google-cloud-java-spring",
         "provider": "Google Cloud"},
    ],
    # ── Django / Flask ──────────────────────────────────────────────────────
    "django": [
        {"title": "Django 4 and Python Full-Stack Developer Masterclass", "platform": "Udemy",
         "url": "https://www.udemy.com/course/django-and-python-full-stack-developer-masterclass/",
         "provider": "Jose Portilla"},
        {"title": "Django for Everybody Specialization", "platform": "Coursera",
         "url": "https://www.coursera.org/specializations/django",
         "provider": "University of Michigan"},
    ],
    "flask": [
        {"title": "REST APIs with Flask and Python", "platform": "Udemy",
         "url": "https://www.udemy.com/course/rest-api-flask-and-python/",
         "provider": "Jose Salvatierra"},
    ],
    # ── Default fallback ────────────────────────────────────────────────────
    "DEFAULT": [
        {"title": "LinkedIn Learning – Skill Development Paths",
         "platform": "LinkedIn Learning",
         "url": "https://www.linkedin.com/learning/",
         "provider": "LinkedIn"},
        {"title": "Coursera – Browse All Courses",
         "platform": "Coursera",
         "url": "https://www.coursera.org/browse",
         "provider": "Coursera"},
        {"title": "Udemy – Top Courses in Every Category",
         "platform": "Udemy",
         "url": "https://www.udemy.com/",
         "provider": "Udemy"},
    ],
}


def search_courses_catalogue(skill: str, max_results: int = 3) -> list[dict]:
    """
    Return curated courses for a skill from the local catalogue.
    Tries exact match first, then partial match on any catalogue key.
    Falls back to DEFAULT entries if no match found.
    No API key required — works completely offline.
    """
    key = skill.strip().lower()

    # Exact match
    if key in COURSE_CATALOGUE:
        return COURSE_CATALOGUE[key][:max_results]

    # Partial match — key is contained in a catalogue entry key or vice-versa
    for cat_key, courses in COURSE_CATALOGUE.items():
        if cat_key == "DEFAULT":
            continue
        if cat_key in key or key in cat_key:
            return courses[:max_results]

    # Word-level partial match
    key_words = set(key.split())
    for cat_key, courses in COURSE_CATALOGUE.items():
        if cat_key == "DEFAULT":
            continue
        cat_words = set(cat_key.split())
        if key_words & cat_words:   # any common word
            return courses[:max_results]

    return COURSE_CATALOGUE["DEFAULT"][:max_results]


def add_course_suggestions(report: GapReport, search_fn=None) -> None:
    """
    Populate course suggestions using the local catalogue (no API needed).
    `search_fn` param kept for backwards compatibility but is ignored.
    """
    for skill in report.missing_core[:3]:
        courses = search_courses_catalogue(skill, max_results=3)
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