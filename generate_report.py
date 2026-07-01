import pandas as pd
import sys
import os
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config import CLEANED_RESUME_DATA_PATH, CLASSIFIER_PATH, VECTORIZER_PATH, REPORT_DIR
from evaluate import plot_confusion_matrix

def generate_final_report():
    print("Generating evaluation reports...")
    
    if not os.path.exists(CLEANED_RESUME_DATA_PATH):
        print("Data not found.")
        return
        
    df = pd.read_csv(CLEANED_RESUME_DATA_PATH)
    with open(CLASSIFIER_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
        
    X_tfidf = vectorizer.transform(df['cleaned_resume'])
    y_true = df['Category']
    y_pred = clf.predict(X_tfidf)
    
    labels = sorted(df['Category'].unique())
    plot_confusion_matrix(y_true, y_pred, labels, filename='resume_classifier_cm.png')
    
    report_text = f"""
# SmartHire Project Evaluation Report

## Resume Classification Model
- **Algorithm**: Logistic Regression
- **Vectorization**: TF-IDF
- **Accuracy**: {clf.score(X_tfidf, y_true):.4f}

## Job Recommendation Engine
- **Method**: Cosine Similarity on TF-IDF vectors
- **Job Corpus Size**: {len(pd.read_csv(os.path.join(os.path.dirname(CLEANED_RESUME_DATA_PATH), 'cleaned_jobs.csv')))}

## Clustering
- **Algorithm**: K-Means
- **Optimal Clusters**: 10
    """
    
    with open(os.path.join(REPORT_DIR, 'evaluation_summary.md'), 'w') as f:
        f.write(report_text)
    
    print(f"Report saved to {REPORT_DIR}")

if __name__ == "__main__":
    generate_final_report()
