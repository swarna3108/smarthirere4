import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import FIGURE_DIR

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', filename='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURE_DIR, filename))
    plt.close()

def get_metrics_summary(y_true, y_pred):
    """Return a dictionary of basic metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'report': classification_report(y_true, y_pred, output_dict=True)
    }
