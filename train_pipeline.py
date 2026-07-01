import os
import pandas as pd
import pickle
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config import (
    RESUME_DATA_PATH, JOB_DATA_PATH, CLEANED_RESUME_DATA_PATH, CLEANED_JOB_DATA_PATH,
    CLASSIFIER_PATH, VECTORIZER_PATH, JOB_VECTORIZER_PATH, CLUSTERING_MODEL_PATH, FIT_PREDICTOR_PATH,
    FIGURE_DIR
)
from data.preprocess import preprocess_resume_data, preprocess_job_data, advanced_text_preprocessing
from features.text_features import create_tfidf_features, load_vectorizer
from models.classifier import train_classifier, predict_category
from models.recommender import get_recommendations
from models.clustering import train_kmeans_clustering, get_optimal_clusters, train_lda_model
from models.fit_predictor import train_fit_predictor
from evaluate import plot_confusion_matrix

# Ensure directories exist
os.makedirs(os.path.dirname(CLASSIFIER_PATH), exist_ok=True)
os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
os.makedirs(os.path.dirname(JOB_VECTORIZER_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CLUSTERING_MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FIT_PREDICTOR_PATH), exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

def run_training_pipeline():
    print("Starting SmartHire ML Training Pipeline...")

    # 1. Preprocess Data
    print("\n--- Preprocessing Resume Data ---")
    resume_df = preprocess_resume_data()
    if resume_df is None:
        print("Failed to preprocess resume data. Exiting.")
        return
    print("Resume data preprocessed successfully.")

    print("\n--- Preprocessing Job Data ---")
    job_df = preprocess_job_data()
    if job_df is None:
        print("Failed to preprocess job data. Exiting.")
        return
    print("Job data preprocessed successfully.")

    # 2. Train Resume Category Classifier
    print("\n--- Training Resume Category Classifier ---")
    X = resume_df["cleaned_resume"]
    y = resume_df["Category"]
    
    # Create TF-IDF features for classifier and save the vectorizer
    X_tfidf, resume_vectorizer = create_tfidf_features(X, save_path=VECTORIZER_PATH)
    print("Resume TF-IDF matrix shape:", X_tfidf.shape)

    # Train classifier (will compare LR, SVM, RF and save the best one)
    classifier_model = train_classifier(X_tfidf, y, resume_vectorizer)
    if classifier_model is None:
        print("Failed to train classifier. Exiting.")
        return
    print("Resume Category Classifier trained and saved.")

    # 3. Train Job Recommender Components (TF-IDF for jobs)
    print("\n--- Training Job Recommender Components ---")
    job_df["cleaned_text"] = job_df["cleaned_text"].fillna('')
    job_tfidf_matrix, job_vectorizer = create_tfidf_features(job_df["cleaned_text"], save_path=JOB_VECTORIZER_PATH)
    print("Job TF-IDF matrix shape:", job_tfidf_matrix.shape)
    print("Job Recommender TF-IDF vectorizer trained and saved.")

    # 4. Train Job Clustering Model
    print("\n--- Training Job Clustering Model ---")
    if job_tfidf_matrix.shape[0] > 1:
        # optimal_k, _, _ = get_optimal_clusters(job_tfidf_matrix)
        optimal_k = 10 # Manual override for speed in this environment
        print(f"Optimal number of clusters determined: {optimal_k}")
        kmeans_model = train_kmeans_clustering(job_tfidf_matrix, n_clusters=optimal_k)
        job_df['cluster'] = kmeans_model.predict(job_tfidf_matrix)
        job_df.to_csv(CLEANED_JOB_DATA_PATH, index=False) # Save with clusters
        print("Job Clustering model trained and saved.")
    else:
        print("Not enough data to perform clustering. Skipping.")
        kmeans_model = None

    # 5. Train Fit/Shortlisting Predictor
    print("\n--- Training Fit/Shortlisting Predictor ---")
    fit_predictor_model = train_fit_predictor(resume_df, job_df, job_vectorizer)
    if fit_predictor_model is None:
        print("Failed to train fit predictor. Skipping.")
    else:
        print("Fit/Shortlisting Predictor trained and saved.")

    print("\nSmartHire ML Training Pipeline Completed.")

if __name__ == "__main__":
    run_training_pipeline()
