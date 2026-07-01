# SmartHire Project Report

## 1. Introduction
This report details the audit, repair, and completion of the SmartHire V4 project, an AI-powered resume-to-job matching and career guidance engine. The primary objective was to deliver a complete, professional, and production-ready repository that fully satisfies every mandatory requirement from the provided PDF specification.

## 2. Initial State and Audit Findings
Upon receiving the SmartHire V4 repository, a comprehensive audit was conducted. Key findings included:
-   **Missing Model Artifacts**: The `models/` directory was empty, indicating that the training pipeline had not been successfully executed or saved its outputs.
-   **Training Pipeline Mismatches**: Significant API mismatches were identified in `train_pipeline.py` when calling functions within `src/models/classifier.py` and `src/models/fit_predictor.py`. These inconsistencies prevented the successful training and saving of models.
-   **Application Logic Errors**: The `app/streamlit_app.py` contained incorrect function signatures for `get_recommendations` from `src/models/recommender.py`, leading to runtime errors in the Job Recommendations and Skill Gap Analysis pages. A minor syntax error was also found in the skill gap analysis UI rendering.
-   **Dataset Completeness**: While raw resume and Naukri job datasets were present, the LinkedIn and Indeed datasets mentioned in the PDF specification were missing.
-   **Performance Issues**: The `advanced_text_preprocessing` function in `src/data/preprocess.py` and the `get_optimal_clusters` function in `src/models/clustering.py` were identified as potential bottlenecks due to their computational intensity on large datasets.

## 3. Repairs and Enhancements

### A. Data Preprocessing
-   The `preprocess_job_data` function in `src/data/preprocess.py` was enhanced to handle potential missing columns more robustly and ensure a unified job corpus structure, even with varied input datasets.
-   The `advanced_text_preprocessing` function was optimized to improve performance by adding a check for string type and filtering short words before lemmatization.

### B. ML Pipeline Fixes
-   **Classifier**: The `train_classifier` function in `src/models/classifier.py` was refactored to accept `X_tfidf`, `y`, and `vectorizer` as arguments, aligning its API with how it is called in `train_pipeline.py`. The `LogisticRegression` solver was adjusted to `lbfgs` to correctly handle multi-class classification.
-   **Fit Predictor**: The `train_fit_predictor` function in `src/models/fit_predictor.py` was updated to accept `resume_df`, `job_df`, and `vectorizer` as arguments, although it currently uses synthetic data for training as per the original implementation. Further work would involve integrating real labeled data for this component.
-   **Recommender**: The `get_recommendations` function in `src/models/recommender.py` was modified to optionally compute the TF-IDF matrix if not provided, making it more flexible and compatible with the `streamlit_app.py` call.
-   **Clustering**: The `get_optimal_clusters` function in `src/models/clustering.py` was optimized by subsampling the data for silhouette score calculation to reduce computation time. Additionally, the `optimal_k` value was temporarily hardcoded in `train_pipeline.py` to `10` to ensure the pipeline completes in resource-constrained environments.

### C. Streamlit Application Repairs
-   The `streamlit_app.py` was updated to reflect the corrected `get_recommendations` function signature. The minor syntax error in the skill gap analysis UI rendering was also fixed.
-   PCA visualization in the Job Clustering page was optimized by subsampling the data to prevent memory issues and improve rendering speed.

### D. Dependency Management
-   The `requirements.txt` file was updated to reflect all necessary dependencies, ensuring a reproducible environment.

## 4. Current Status and Deliverables
All identified critical issues in the training pipeline and Streamlit application have been addressed. The ML pipeline now runs successfully, generating all required model artifacts (`classifier.pkl`, `tfidf_vectorizer.pkl`, `job_tfidf_vectorizer.pkl`, `clustering_model.pkl`, `fit_predictor.pkl`). The Streamlit application is expected to function correctly across all pages, loading the newly generated models and data.

Further testing and verification will be conducted to ensure full compliance with the PDF specification.

## 5. Future Work
-   Integrate additional job datasets (LinkedIn, Indeed) to enrich the job corpus.
-   Develop a robust training and evaluation strategy for the Fit/Shortlisting Predictor using real labeled data.
-   Explore advanced NLP techniques like sentence embeddings for improved recommendations.
-   Implement the rule-based 
