import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLEANED_JOB_DATA_PATH, JOB_VECTORIZER_PATH
from features.text_features import create_tfidf_features, load_vectorizer

def build_recommender():
    """Build the job recommender engine."""
    if not os.path.exists(CLEANED_JOB_DATA_PATH):
        print("Cleaned job data not found. Please run preprocessing first.")
        return None
    
    df = pd.read_csv(CLEANED_JOB_DATA_PATH)
    # Ensure no NaN in cleaned_text
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    # Create features
    tfidf_matrix, vectorizer = create_tfidf_features(df['cleaned_text'], save_path=JOB_VECTORIZER_PATH)
    
    return tfidf_matrix, vectorizer, df

def get_recommendations(resume_text, vectorizer, df, tfidf_matrix=None, top_n=10):
    """Get top N job recommendations for a resume."""
    if tfidf_matrix is None:
        # If matrix is not provided, compute it from the dataframe
        df['cleaned_text'] = df['cleaned_text'].fillna('')
        tfidf_matrix = vectorizer.transform(df['cleaned_text'])
        
    resume_tfidf = vectorizer.transform([resume_text])
    cosine_sim = cosine_similarity(resume_tfidf, tfidf_matrix)
    
    # Get indices of top N matches
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[:top_n]]
    top_scores = [i[1] for i in sim_scores[:top_n]]
    
    recommendations = df.iloc[top_indices].copy()
    recommendations['match_score'] = top_scores
    
    return recommendations

def skill_gap_analysis(resume_text, job_skills):
    """Analyze skill gap between resume and job requirements."""
    # Simple rule-based skill extraction for demonstration
    # In a real project, this would use a predefined skill list or NER
    resume_words = set(resume_text.lower().split())
    
    if isinstance(job_skills, str):
        job_skills_list = [s.strip().lower() for s in job_skills.split(',')]
    else:
        job_skills_list = []
        
    found_skills = [s for s in job_skills_list if any(word in resume_words for word in s.split())]
    missing_skills = [s for s in job_skills_list if s not in found_skills]
    
    return found_skills, missing_skills
