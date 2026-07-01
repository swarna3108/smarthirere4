import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_skills(text):
    """Extracts skills from a given text using a predefined list of common skills.
    This is a simplified version. For a real-world application, a more sophisticated NLP approach
    with a comprehensive skill taxonomy would be required.
    """
    # A very basic list of skills. In a real project, this would be much more extensive
    # and potentially loaded from a database or external file.
    common_skills = [
        "python", "java", "c++", "javascript", "html", "css", "react", "angular",
        "node.js", "sql", "nosql", "aws", "azure", "gcp", "docker", "kubernetes",
        "machine learning", "deep learning", "data science", "nlp", "computer vision",
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
        "excel", "power bi", "tableau", "r", "spark", "hadoop", "kafka",
        "git", "jira", "agile", "scrum", "communication", "teamwork", "leadership",
        "project management", "financial analysis", "marketing", "sales", "hr",
        "cloud computing", "cybersecurity", "devops", "web development", "mobile development"
    ]
    
    text_lower = text.lower()
    found_skills = [skill for skill in common_skills if skill in text_lower]
    return found_skills

def calculate_skill_overlap(resume_skills_str, job_skills_str):
    """Calculates the percentage of overlap between skills in a resume and job description.
    Assumes skills are comma-separated strings or can be extracted.
    """
    if not isinstance(resume_skills_str, str) or not isinstance(job_skills_str, str):
        return 0.0

    resume_skills = set([s.strip().lower() for s in resume_skills_str.split(",") if s.strip()])
    job_skills = set([s.strip().lower() for s in job_skills_str.split(",") if s.strip()])

    if not job_skills:
        return 0.0

    common_skills = resume_skills.intersection(job_skills)
    return len(common_skills) / len(job_skills)

def calculate_experience_match(resume_experience_str, job_experience_str):
    """Calculates a simple experience match score.
    This is a heuristic. A more advanced system would parse years of experience more robustly.
    """
    resume_exp_years = 0
    job_exp_years = 0

    # Extract numbers from experience strings
    resume_match = re.search(r"\d+", str(resume_experience_str))
    job_match = re.search(r"\d+", str(job_experience_str))

    if resume_match: resume_exp_years = int(resume_match.group())
    if job_match: job_exp_years = int(job_match.group())

    if job_exp_years == 0: # If job has no experience requirement, any experience is a match
        return 1.0
    
    # Simple match: if resume experience meets or exceeds job experience
    return 1.0 if resume_exp_years >= job_exp_years else (resume_exp_years / job_exp_years if job_exp_years > 0 else 0.0)

def get_missing_and_recommended_skills(resume_text, job_description_text):
    """Identifies missing and recommended skills based on resume and job description.
    """
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description_text))

    missing_skills = list(job_skills - resume_skills)
    recommended_skills = list(job_skills - resume_skills) # For now, same as missing, can be refined
    found_skills = list(resume_skills.intersection(job_skills))

    return found_skills, missing_skills, recommended_skills
