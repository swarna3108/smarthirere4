import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
INTERIM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'interim')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Model directory
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Report directory
REPORT_DIR = os.path.join(BASE_DIR, 'reports')
FIGURE_DIR = os.path.join(REPORT_DIR, 'figures')

# Dataset paths
RESUME_DATA_PATH = os.path.join(RAW_DATA_DIR, 'UpdatedResumeDataSet.csv')
JOB_DATA_PATH = os.path.join(RAW_DATA_DIR, 'naukri_com-job_sample.csv')

# Preprocessing parameters
CLEANED_RESUME_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'cleaned_resumes.csv')
CLEANED_JOB_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'cleaned_jobs.csv')

# Model paths
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'classifier.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
FIT_PREDICTOR_PATH = os.path.join(MODEL_DIR, 'fit_predictor.pkl')
CLUSTERING_MODEL_PATH = os.path.join(MODEL_DIR, 'clustering_model.pkl')
JOB_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'job_tfidf_vectorizer.pkl')
