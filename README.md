# SmartHire - Resume-to-Job Matching & Career Guidance Engine

This project implements SmartHire, an AI-powered portal designed to assist job seekers by matching their resumes with suitable job openings, providing fit/shortlisting scores, and generating skill-gap reports. The system is built using classical machine learning techniques, focusing on practical applications without relying on large language models or generative AI.

## Project Features

-   **Resume Upload (PDF, DOCX, TXT)**: Allows users to upload their resumes in various formats.
-   **Resume Text Extraction**: Extracts plain text content from uploaded resumes.
-   **Text Preprocessing**: Cleans and prepares text data for machine learning models.
-   **Resume Category Classifier**: Categorizes resumes into predefined job domains using TF-IDF and Logistic Regression, with comparisons to SVM and Random Forest.
-   **Job Recommendation Engine**: Recommends top N matching jobs based on resume-job similarity using TF-IDF and Cosine Similarity.
-   **Skill Gap Analysis**: Identifies missing skills in a candidate's resume for target job roles.
-   **Job Clustering**: Groups similar job postings using K-Means clustering, with optimal 'k' determined by Elbow Method and Silhouette Score.
-   **Fit/Shortlisting Predictor**: Predicts the likelihood of a resume being shortlisted for a specific job using Logistic Regression.
-   **Model Evaluation**: Comprehensive evaluation of all models using metrics like Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
-   **Streamlit Web Application**: An interactive and modern web interface for users to interact with the SmartHire system.
-   **Modular Python Code**: Well-organized and maintainable codebase.
-   **Logging and Error Handling**: Robust logging and error handling mechanisms.

## Repository Structure

```
smarthire/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # Original downloaded datasets (never edit)
│   ├── interim/              # Merged / partially cleaned data
│   └── processed/            # Final model-ready data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_resume_classifier.ipynb
│   ├── 03_recommender.ipynb
│   ├── 04_clustering_topics.ipynb
│   └── 05_fit_predictor.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── text_features.py
│   │   └── match_features.py
│   ├── models/
│   │   ├── classifier.py
│   │   ├── recommender.py
│   │   ├── clustering.py
│   │   └── fit_predictor.py
│   ├── parsing/
│   │   └── resume_parser.py
│   └── evaluate.py
├── models/                   # Saved .pkl model files
│   ├── classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── job_tfidf_vectorizer.pkl
│   ├── clustering_model.pkl
│   └── fit_predictor.pkl
├── app/
│   └── streamlit_app.py
├── reports/
│   ├── figures/              # PCA/t-SNE plots, confusion matrices
│   └── final_report.pdf      # Written project report
├── train_pipeline.py         # Script to run the full ML training pipeline
├── generate_report.py        # Script to generate project reports
└── SETUP_GUIDE.md            # Setup and installation guide
```

## Setup and Run

To set up and run the SmartHire project, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd smarthire
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK data** (if not already downloaded by the pipeline):
    ```bash
    python -m nltk.downloader stopwords wordnet
    ```

4.  **Train the models**:
    ```bash
    python train_pipeline.py
    ```

5.  **Run the Streamlit application**:
    ```bash
    streamlit run app/streamlit_app.py
    ```

    The application will open in your web browser.

## Datasets

The project currently uses the following datasets, which should be placed in the `data/raw/` directory:

-   `UpdatedResumeDataSet.csv`: Contains resume texts labeled into 25 job categories.
-   `naukri_com-job_sample.csv`: Contains job postings from Naukri.com.

These datasets are automatically preprocessed and utilized by the system. If you wish to use additional job datasets (e.g., LinkedIn, Indeed), they should be placed in the `data/raw/` directory and `src/data/preprocess.py` might need adjustments to merge them into the unified job corpus.

## Deployment

This project is designed to be self-contained and easily deployable to platforms like Streamlit Community Cloud, Render, or Hugging Face Spaces.

1.  Ensure all dependencies are listed in `requirements.txt`.
2.  Push the repository to GitHub.
3.  Connect your GitHub repository to your chosen hosting platform.
4.  Set the main file path to `app/streamlit_app.py`.
5.  Deploy!

## Verification Checklist

Before considering the project complete, ensure all mandatory requirements from the project PDF have been implemented, tested, and verified. A detailed checklist will be provided as a final deliverable.

## Future Improvements

-   Integration with live job board APIs (e.g., LinkedIn, Indeed).
-   Advanced NLP using sentence embeddings (Word2Vec, GloVe, BERT).
-   Implementation of a learning-to-rank model for better job recommendations.
-   A rule-based "mentor" chatbot to answer skill-related questions.

## License

This project is for educational purposes.
