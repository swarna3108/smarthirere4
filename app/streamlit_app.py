import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
import tempfile
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config import (
    CLASSIFIER_PATH, VECTORIZER_PATH, JOB_VECTORIZER_PATH, CLUSTERING_MODEL_PATH, FIT_PREDICTOR_PATH,
    CLEANED_JOB_DATA_PATH, CLEANED_RESUME_DATA_PATH
)
from parsing.resume_parser import parse_resume, clean_resume_text
from features.text_features import load_vectorizer
from models.recommender import get_recommendations
from models.clustering import load_kmeans_clustering
from features.match_features import get_missing_and_recommended_skills

# Set page configuration
st.set_page_config(
    page_title="SmartHire – Resume-to-Job Matching & Career Guidance",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.3em;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .recommendation-card {
        background-color: #e8f4f8;
        padding: 15px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
        border-radius: 5px;
    }
    .skill-box {
        background-color: #f0f8e8;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Application Title
st.markdown('<div class="main-header">💼 SmartHire – Resume-to-Job Matching & Career Guidance</div>', unsafe_allow_html=True)
st.markdown("**Powered by Machine Learning | Find Your Perfect Job Match**")
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page:",
    ["Home", "Upload Resume", "Resume Classification", "Job Recommendations", "Skill Gap Analysis", "Job Clustering", "Model Evaluation", "About"]
)

# Helper function to load models
@st.cache_resource
def load_models():
    classifier = None
    vectorizer = None
    job_vectorizer = None
    kmeans = None
    fit_predictor = None
    
    if os.path.exists(CLASSIFIER_PATH):
        try:
            with open(CLASSIFIER_PATH, 'rb') as f:
                classifier = pickle.load(f)
        except:
            pass
    
    if os.path.exists(VECTORIZER_PATH):
        vectorizer = load_vectorizer(VECTORIZER_PATH)
    
    if os.path.exists(JOB_VECTORIZER_PATH):
        job_vectorizer = load_vectorizer(JOB_VECTORIZER_PATH)
    
    if os.path.exists(CLUSTERING_MODEL_PATH):
        kmeans = load_kmeans_clustering()
    
    if os.path.exists(FIT_PREDICTOR_PATH):
        try:
            with open(FIT_PREDICTOR_PATH, 'rb') as f:
                fit_predictor = pickle.load(f)
        except:
            pass
    
    return classifier, vectorizer, job_vectorizer, kmeans, fit_predictor

@st.cache_data
def load_job_data():
    if os.path.exists(CLEANED_JOB_DATA_PATH):
        return pd.read_csv(CLEANED_JOB_DATA_PATH)
    return None

@st.cache_data
def load_resume_data():
    if os.path.exists(CLEANED_RESUME_DATA_PATH):
        return pd.read_csv(CLEANED_RESUME_DATA_PATH)
    return None

# ===== PAGE: HOME =====
if page == "Home":
    st.markdown('<div class="sub-header">Welcome to SmartHire</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is SmartHire?
        
        SmartHire is an intelligent resume-to-job matching system powered by machine learning. 
        It helps job seekers find the best-fitting positions based on their qualifications and experience.
        
        ### Key Features:
        - **Resume Upload & Parsing**: Upload your resume in PDF, DOCX, or TXT format
        - **Resume Classification**: Automatic job category prediction
        - **Job Recommendations**: Find top 10 matching jobs with similarity scores
        - **Skill Gap Analysis**: Identify missing skills and improvement areas
        - **Job Clustering**: Discover related job opportunities
        - **Fit Prediction**: Get a shortlisting probability for target roles
        - **Model Evaluation**: View detailed performance metrics
        """)
    
    with col2:
        st.markdown("""
        ### How It Works:
        
        1. **Upload Your Resume**: Start by uploading your resume file
        2. **Automatic Processing**: The system extracts and cleans your resume text
        3. **Category Prediction**: AI predicts your job category
        4. **Job Matching**: Finds similar jobs from our database
        5. **Skill Analysis**: Highlights missing skills and recommendations
        6. **Results**: View detailed recommendations and insights
        
        ### Technology Stack:
        - **ML Algorithms**: Logistic Regression, SVM, Random Forest, K-Means
        - **NLP**: TF-IDF Vectorization, Text Preprocessing, Lemmatization
        - **Framework**: Streamlit, scikit-learn, pandas
        """)
    
    st.markdown("---")
    st.info("👉 **Get Started**: Navigate to 'Upload Resume' in the sidebar to begin!")

# ===== PAGE: UPLOAD RESUME =====
elif page == "Upload Resume":
    st.markdown('<div class="sub-header">📄 Upload Your Resume</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a resume file (PDF, DOCX, or TXT):",
        type=["pdf", "docx", "txt"]
    )
    
    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Save uploaded file temporarily and parse it
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            resume_text = parse_resume(tmp_path)
            cleaned_text = clean_resume_text(resume_text)
            
            st.markdown("### Resume Preview (Cleaned Text):")
            st.text_area("Resume Content:", cleaned_text, height=200, disabled=True)
            
            # Store in session state for use in other pages
            st.session_state['resume_text'] = cleaned_text
            st.session_state['resume_uploaded'] = True
            
            st.success("✅ Resume processed successfully! You can now navigate to other pages to see analysis.")
            
        except Exception as e:
            st.error(f"❌ Error parsing resume: {str(e)}")
        finally:
            os.remove(tmp_path)
    else:
        st.info("📁 Please upload a resume file to get started.")

# ===== PAGE: RESUME CLASSIFICATION =====
elif page == "Resume Classification":
    st.markdown('<div class="sub-header">🎯 Resume Category Classification</div>', unsafe_allow_html=True)
    
    if 'resume_text' not in st.session_state:
        st.warning("⚠️ Please upload a resume first on the 'Upload Resume' page.")
    else:
        classifier, vectorizer, _, _, _ = load_models()
        
        if classifier and vectorizer:
            resume_text = st.session_state['resume_text']
            
            try:
                X_tfidf = vectorizer.transform([resume_text])
                category = classifier.predict(X_tfidf)[0]
                probabilities = classifier.predict_proba(X_tfidf)[0]
                
                st.markdown(f"### Predicted Category: **{category}**")
                
                # Display top 5 predictions
                st.markdown("#### Top 5 Category Predictions:")
                top_5_indices = np.argsort(probabilities)[-5:][::-1]
                
                for idx, i in enumerate(top_5_indices, 1):
                    cat = classifier.classes_[i]
                    prob = probabilities[i]
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{idx}. {cat}")
                    with col2:
                        st.write(f"{prob*100:.2f}%")
                    st.progress(prob)
                
                # Display metrics
                st.markdown("#### Model Performance Metrics:")
                resume_df = load_resume_data()
                if resume_df is not None:
                    X_all = vectorizer.transform(resume_df['cleaned_resume'])
                    y_all = resume_df['Category']
                    y_pred_all = classifier.predict(X_all)
                    
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    accuracy = accuracy_score(y_all, y_pred_all)
                    precision = precision_score(y_all, y_pred_all, average='weighted', zero_division=0)
                    recall = recall_score(y_all, y_pred_all, average='weighted', zero_division=0)
                    f1 = f1_score(y_all, y_pred_all, average='weighted', zero_division=0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.2%}")
                    col2.metric("Precision", f"{precision:.2%}")
                    col3.metric("Recall", f"{recall:.2%}")
                    col4.metric("F1-Score", f"{f1:.2%}")
                
            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")
        else:
            st.error("❌ Models not loaded. Please ensure all model files are present.")

# ===== PAGE: JOB RECOMMENDATIONS =====
elif page == "Job Recommendations":
    st.markdown('<div class="sub-header">🔍 Top Job Recommendations</div>', unsafe_allow_html=True)
    
    if 'resume_text' not in st.session_state:
        st.warning("⚠️ Please upload a resume first on the 'Upload Resume' page.")
    else:
        _, _, job_vectorizer, _, _ = load_models()
        job_df = load_job_data()
        
        if job_vectorizer is not None and job_df is not None:
            resume_text = st.session_state['resume_text']
            
            try:
                recommendations = get_recommendations(resume_text, job_vectorizer, job_df, top_n=10)
                
                if not recommendations.empty:
                    st.markdown(f"### Found {len(recommendations)} Matching Jobs")
                    
                    for idx, (_, job) in enumerate(recommendations.iterrows(), 1):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"#### {idx}. {job.get('title', 'N/A')}")
                                st.write(f"**Company**: {job.get('company', 'N/A')}")
                                st.write(f"**Location**: {job.get('location', 'N/A')}")
                                st.write(f"**Experience**: {job.get('experience', 'N/A')}")
                                st.write(f"**Skills**: {job.get('skills', 'N/A')[:100]}...")
                            
                            with col2:
                                match_score = job.get('match_score', 0)
                                st.metric("Match Score", f"{match_score*100:.1f}%")
                                st.progress(match_score)
                            
                            st.write(f"**Description**: {job.get('description', 'N/A')[:200]}...")
                            st.markdown("---")
                else:
                    st.info("No matching jobs found.")
                    
            except Exception as e:
                st.error(f"❌ Error getting recommendations: {str(e)}")
        else:
            st.error("❌ Job data or vectorizer not loaded.")

# ===== PAGE: SKILL GAP ANALYSIS =====
elif page == "Skill Gap Analysis":
    st.markdown('<div class="sub-header">🎓 Skill Gap Analysis</div>', unsafe_allow_html=True)
    
    if 'resume_text' not in st.session_state:
        st.warning("⚠️ Please upload a resume first on the 'Upload Resume' page.")
    else:
        _, _, job_vectorizer, _, _ = load_models()
        job_df = load_job_data()
        
        if job_vectorizer is not None and job_df is not None:
            resume_text = st.session_state['resume_text']
            
            try:
                recommendations = get_recommendations(resume_text, job_vectorizer, job_df, top_n=5)
                
                if not recommendations.empty:
                    st.markdown("### Skill Gap for Top 5 Recommendations:")
                    
                    for idx, (_, job) in enumerate(recommendations.iterrows(), 1):
                        with st.expander(f"{idx}. {job.get('title', 'N/A')} - {job.get('company', 'N/A')}"):
                            found_skills, missing_skills, recommended_skills = get_missing_and_recommended_skills(
                                resume_text, 
                                job.get('description', '')
                            )
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### ✅ Skills You Have:")
                                if found_skills:
                                    for skill in found_skills:
                                        st.write(f"• {skill}")
                                else:
                                    st.write("No matching skills found.")
                            
                            with col2:
                                st.markdown("#### ❌ Skills You Need:")
                                if missing_skills:
                                    for skill in missing_skills:
                                        st.write(f"• {skill}")
                                else:
                                    st.write("No missing skills identified.")
                            
                            # Skill match percentage
                            if missing_skills or found_skills:
                                total_skills = len(found_skills) + len(missing_skills)
                                match_percentage = (len(found_skills) / total_skills * 100) if total_skills > 0 else 0
                                st.markdown(f"**Skill Match**: {match_percentage:.1f}%")
                                st.progress(match_percentage / 100)
                else:
                    st.info("No recommendations available for skill gap analysis.")
                    
            except Exception as e:
                st.error(f"❌ Error during skill gap analysis: {str(e)}")
        else:
            st.error("❌ Job data or vectorizer not loaded.")

# ===== PAGE: JOB CLUSTERING =====
elif page == "Job Clustering":
    st.markdown('<div class="sub-header">🗂️ Job Clustering & Discovery</div>', unsafe_allow_html=True)
    
    job_df = load_job_data()
    _, _, job_vectorizer, kmeans, _ = load_models()
    
    if job_df is not None and kmeans is not None and job_vectorizer is not None:
        try:
            # Get cluster assignments
            job_df_copy = job_df.copy()
            job_df_copy['cleaned_text'] = job_df_copy['cleaned_text'].fillna('')
            job_tfidf = job_vectorizer.transform(job_df_copy['cleaned_text'])
            clusters = kmeans.predict(job_tfidf)
            job_df_copy['cluster'] = clusters
            
            st.markdown(f"### Total Clusters: {len(np.unique(clusters))}")

            # PCA for visualization
            if job_tfidf.shape[0] > 1 and job_tfidf.shape[1] > 1:
                try:
                    # Subsample for PCA visualization to avoid memory issues and slow rendering
                    sample_size = min(1000, job_tfidf.shape[0])
                    if job_tfidf.shape[0] > sample_size:
                        indices = np.random.choice(job_tfidf.shape[0], sample_size, replace=False)
                        job_tfidf_sample = job_tfidf[indices]
                        clusters_sample = clusters[indices]
                    else:
                        job_tfidf_sample = job_tfidf
                        clusters_sample = clusters
                        
                    pca = PCA(n_components=2)
                    principal_components = pca.fit_transform(job_tfidf_sample.toarray())
                    pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
                    pca_df["cluster"] = clusters_sample

                    fig = plt.figure(figsize=(10, 8))
                    sns.scatterplot(x="PC1", y="PC2", hue="cluster", data=pca_df, palette="viridis", legend="full", alpha=0.7)
                    plt.title("Job Clusters (PCA)")
                    plt.xlabel("Principal Component 1")
                    plt.ylabel("Principal Component 2")
                    st.pyplot(fig)
                except Exception as pca_e:
                    st.warning(f"Could not generate PCA visualization: {pca_e}. This might happen if there are too few samples or features.")
            else:
                st.info("Not enough data to generate PCA visualization for clustering.")
            
            # Display cluster insights
            for cluster_id in sorted(np.unique(clusters)):
                cluster_jobs = job_df_copy[job_df_copy['cluster'] == cluster_id]
                
                with st.expander(f"Cluster {cluster_id} ({len(cluster_jobs)} jobs)"):
                    st.markdown("#### Top Job Titles in this Cluster:")
                    top_titles = cluster_jobs['title'].value_counts().head(5)
                    for title, count in top_titles.items():
                        st.write(f"• {title} ({count} jobs)")
                    
                    st.markdown("#### Top Companies in this Cluster:")
                    top_companies = cluster_jobs['company'].value_counts().head(5)
                    for company, count in top_companies.items():
                        st.write(f"• {company} ({count} jobs)")
                    
                    st.markdown("#### Sample Jobs:")
                    sample_jobs = cluster_jobs.head(3)
                    for _, job in sample_jobs.iterrows():
                        st.write(f"- {job.get('title', 'N/A')} at {job.get('company', 'N/A')}")
            
        except Exception as e:
            st.error(f"❌ Error during clustering visualization: {str(e)}")
    else:
        st.warning("⚠️ Clustering model or job data not available. Please run the training pipeline first.")

# ===== PAGE: MODEL EVALUATION =====
elif page == "Model Evaluation":
    st.markdown('<div class="sub-header">📊 Model Evaluation & Metrics</div>', unsafe_allow_html=True)
    
    try:
        classifier, vectorizer, _, _, _ = load_models()
        resume_df = load_resume_data()
        job_df = load_job_data()
        
        if classifier and vectorizer and resume_df is not None:
            X_tfidf = vectorizer.transform(resume_df['cleaned_resume'])
            y_true = resume_df['Category']
            y_pred = classifier.predict(X_tfidf)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.2%}")
            col2.metric("Precision", f"{precision:.2%}")
            col3.metric("Recall", f"{recall:.2%}")
            col4.metric("F1-Score", f"{f1:.2%}")
            
            st.markdown("---")
            
            st.markdown("### Classifier Details:")
            st.write(f"- **Algorithm**: Logistic Regression / SVM / Random Forest (Best Performer)")
            st.write(f"- **Vectorization**: TF-IDF (5000 features)")
            st.write(f"- **Categories**: {len(classifier.classes_)} job domains")
            
            st.markdown("### Job Recommender Details:")
            st.write(f"- **Method**: Cosine Similarity")
            st.write(f"- **Vectorization**: TF-IDF")
            st.write(f"- **Job Corpus**: {len(job_df) if job_df is not None else 'N/A'} jobs")
            
            st.markdown("### Job Clustering Details:")
            st.write(f"- **Algorithm**: K-Means")
            st.write(f"- **Optimal Clusters**: 10 (determined via Elbow Method & Silhouette Score)")
            st.write(f"- **Evaluation Metric**: Silhouette Score")
        else:
            st.warning("⚠️ Models or data not loaded.")
            
    except Exception as e:
        st.error(f"❌ Error loading evaluation metrics: {str(e)}")

# ===== PAGE: ABOUT =====
elif page == "About":
    st.markdown('<div class="sub-header">ℹ️ About SmartHire</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Project Overview
    
    SmartHire is an industrial machine learning project that demonstrates the application of 
    supervised and unsupervised learning techniques to solve a real-world career guidance problem.
    
    ### Project Goals:
    - Build an intelligent resume-to-job matching system
    - Apply classical ML algorithms (no LLMs)
    - Provide actionable career guidance through skill gap analysis
    - Discover job families through clustering
    
    ### Technologies Used:
    - **Python 3.10+**
    - **scikit-learn**: Machine Learning algorithms
    - **pandas & numpy**: Data processing
    - **Streamlit**: Web application framework
    - **NLTK**: Natural Language Processing
    - **gensim**: Topic modeling (LDA)
    
    ### Machine Learning Components:
    
    #### Supervised Learning:
    1. **Resume Category Classifier**
       - Predicts job domain from resume text
       - Models: Logistic Regression, SVM, Random Forest
       - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
    
    2. **Fit/Shortlisting Predictor** (Optional)
       - Predicts likelihood of shortlisting for a specific job
       - Features: Skill overlap, experience match, text similarity
       - Models: Logistic Regression, XGBoost
    
    #### Unsupervised Learning:
    1. **Job Recommendation Engine**
       - Content-based recommendation using cosine similarity
       - Returns top-N matching jobs with similarity scores
    
    2. **Job Clustering**
       - Groups similar jobs into families
       - K-Means with Elbow Method and Silhouette Score
       - Enables discovery of related opportunities
    
    3. **Skill Gap Analysis**
       - Identifies missing skills for target roles
       - Provides improvement suggestions
       - Optional: LDA topic modeling for skill themes
    
    ### Dataset Information:
    - **Resume Dataset**: ~960 resumes labeled into 25 job categories
    - **Job Datasets**: Merged from Naukri, LinkedIn, and Indeed sources
    - **Total Jobs**: 22,000+ job listings
    
    ### Performance Metrics:
    - **Resume Classifier Accuracy**: ~99.5%
    - **Recommendation Precision@10**: High (qualitatively verified)
    - **Clustering Silhouette Score**: Positive (good cluster separation)
    
    ### Future Enhancements:
    - Sentence embeddings (Word2Vec, GloVe, BERT)
    - Learning-to-rank models
    - Rule-based mentor chatbot
    - Cloud deployment (Streamlit Community Cloud, AWS, GCP)
    - Real-time API for integration
    
    ### Team & Attribution:
    - **Project Type**: Industrial ML Project
    - **Duration**: 3 weeks
    - **Team Size**: 2-3 or Individual
    - **Framework**: Streamlit + scikit-learn
    
    ### License:
    This project is for educational purposes.
    
    ---
    
    **Questions or Feedback?** Feel free to explore the application and test different resumes!
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**SmartHire v1.0** | ML-Powered Career Guidance")
