from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTORIZER_PATH, JOB_VECTORIZER_PATH

def create_tfidf_features(texts, max_features=5000, save_path=None):
    """Create TF-IDF features from a list of texts."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"Vectorizer saved to {save_path}")
        
    return tfidf_matrix, vectorizer

def load_vectorizer(path):
    """Load a saved TF-IDF vectorizer."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def transform_text(text, vectorizer):
    """Transform text using a loaded vectorizer."""
    return vectorizer.transform([text])
