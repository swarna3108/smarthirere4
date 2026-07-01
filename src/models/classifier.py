import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLEANED_RESUME_DATA_PATH, CLASSIFIER_PATH, VECTORIZER_PATH
from features.text_features import create_tfidf_features

def train_classifier(X_tfidf, y, vectorizer):
    """Train the resume category classifier."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
        "SVM": SVC(kernel='linear', random_state=42, probability=True),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    # Save the best performing model
    os.makedirs(os.path.dirname(CLASSIFIER_PATH), exist_ok=True)
    with open(CLASSIFIER_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nBest classifier ({best_model_name}) saved to {CLASSIFIER_PATH}")
    
    return best_model

def predict_category(text, clf, vectorizer):
    """Predict the category of a resume."""
    X_tfidf = vectorizer.transform([text])
    prediction = clf.predict(X_tfidf)
    return prediction[0]

if __name__ == "__main__":
    train_classifier()
