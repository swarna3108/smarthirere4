import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import os

from config import CLUSTERING_MODEL_PATH, FIGURE_DIR

def train_kmeans_clustering(tfidf_matrix, n_clusters=10):
    """Trains a K-Means clustering model and saves it.

    Args:
        tfidf_matrix: TF-IDF matrix of job descriptions.
        n_clusters (int): Number of clusters to form.

    Returns:
        KMeans: Trained K-Means model.
    """
    print(f"Training K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)

    os.makedirs(os.path.dirname(CLUSTERING_MODEL_PATH), exist_ok=True)
    with open(CLUSTERING_MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"K-Means clustering model saved to {CLUSTERING_MODEL_PATH}")
    return kmeans

def load_kmeans_clustering():
    """Loads a pre-trained K-Means clustering model."""
    if os.path.exists(CLUSTERING_MODEL_PATH):
        with open(CLUSTERING_MODEL_PATH, 'rb') as f:
            kmeans = pickle.load(f)
        print(f"K-Means clustering model loaded from {CLUSTERING_MODEL_PATH}")
        return kmeans
    else:
        print(f"K-Means clustering model not found at {CLUSTERING_MODEL_PATH}")
        return None

def get_optimal_clusters(tfidf_matrix, k_range=range(2, 11)):
    """Determines the optimal number of clusters using Elbow Method and Silhouette Score.

    Args:
        tfidf_matrix: TF-IDF matrix of job descriptions.
        k_range (range): Range of k values to test.

    Returns:
        tuple: (optimal_k, sse_scores, silhouette_scores)
    """
    sse = []
    silhouette_scores = []
    
    # Subsample for silhouette score if dataset is large
    sample_size = min(2000, tfidf_matrix.shape[0])
    if tfidf_matrix.shape[0] > sample_size:
        indices = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
        tfidf_sample = tfidf_matrix[indices]
    else:
        tfidf_sample = tfidf_matrix

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        kmeans.fit(tfidf_matrix)
        sse.append(kmeans.inertia_)
        
        if tfidf_sample.shape[0] > 1:
            sample_labels = kmeans.predict(tfidf_sample)
            if len(np.unique(sample_labels)) > 1:
                silhouette_scores.append(silhouette_score(tfidf_sample, sample_labels))
            else:
                silhouette_scores.append(0)
        else:
            silhouette_scores.append(0)
            
    # Simple heuristic to find optimal k (e.g., max silhouette score)
    if silhouette_scores:
        # Ensure k_range and silhouette_scores have the same length for argmax
        if len(k_range) == len(silhouette_scores):
            optimal_k = k_range[np.argmax(silhouette_scores)]
        else:
            # Fallback if lengths don't match (e.g., due to filtering for silhouette)
            optimal_k = 10 # Default if calculation is problematic
    else:
        optimal_k = 10 # Default if no scores are valid

    return optimal_k, sse, silhouette_scores

# Optional: LDA Topic Modeling (as mentioned in PDF for skill gap insight)
from gensim.models import LdaModel
from gensim.corpora import Dictionary

def train_lda_model(texts, num_topics=5):
    """Trains an LDA topic model.

    Args:
        texts (list of list of str): Tokenized texts.
        num_topics (int): Number of topics to extract.

    Returns:
        LdaModel: Trained LDA model.
        Dictionary: Gensim Dictionary.
    """
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    print(f"Training LDA model with {num_topics} topics...")
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)
    print("LDA model trained.")
    return lda_model, dictionary

def get_topic_keywords(lda_model, num_words=5):
    """Extracts keywords for each topic from an LDA model."""
    topics = {}
    for idx, topic in lda_model.print_topics(-1):
        topics[idx] = [word.split("*")[1].replace("\"", "").strip() for word in topic.split("+")]
    return topics
