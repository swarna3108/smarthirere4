# SmartHire Setup and Usage Guide

This guide provides detailed instructions on how to set up and run the SmartHire project on your local machine.

## 1. Prerequisites

*   **Python 3.7 or higher**: Ensure Python is installed and accessible via the `python` command in your terminal.
*   **pip**: Python's package installer, usually comes with Python.
*   **Virtual environment (recommended)**: To manage project dependencies in an isolated environment.

## 2. Installation Steps

### 2.1. Clone or Extract the Repository

First, navigate to your desired directory and clone the repository or extract the provided ZIP file:

```bash
# If cloning from Git
git clone <repository_url>
cd smarthire

# If extracting a ZIP file
# Ensure you are in the 'smarthire' directory after extraction
cd smarthire
```

### 2.2. Create a Virtual Environment (Optional but Recommended)

It is highly recommended to create and activate a virtual environment to avoid conflicts with other Python projects:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2.3. Install Dependencies

Install all required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2.4. Download NLTK Data

The project uses NLTK for text preprocessing. Download the necessary NLTK data (stopwords and wordnet):

```bash
python -m nltk.downloader stopwords wordnet
```

### 2.5. Prepare Datasets

The project requires the following datasets to be present in the `data/raw/` directory:

*   `UpdatedResumeDataSet.csv`: Contains resume texts labeled into 25 job categories.
*   `naukri_com-job_sample.csv`: Contains job postings from Naukri.com.

These datasets will be automatically preprocessed and utilized by the system. If you use different datasets, ensure they match the expected schema or modify the `src/data/load_data.py` and `src/data/preprocess.py` scripts accordingly.

### 2.6. Train the Models (REQUIRED FIRST STEP)

To ensure the machine learning models are compatible with your local Python and library versions (especially `scikit-learn`), you **must** run the training pipeline on your machine before starting the Streamlit application. This step will preprocess data, train all ML models, and save them to the `models/` directory.

```bash
python train_pipeline.py
```

This script will:
*   Preprocess the resume and job datasets.
*   Train the resume category classifier (comparing Logistic Regression, SVM, and Random Forest).
*   Build components for the job recommender engine.
*   Perform job clustering (with optimal `k` currently hardcoded to 10 for faster execution in this environment).
*   Train the fit/shortlisting predictor.

### 2.7. Run the Streamlit Application

Once the models are trained, you can launch the Streamlit web application:

```bash
streamlit run app/streamlit_app.py
```

The application will open in your default web browser, typically at `http://localhost:8501`.

## 3. Running Unit Tests

To run the basic unit tests for core functionalities:

```bash
python tests/test_features.py
```

## 4. Troubleshooting

*   **`Python was not found` or `python3` command not working**: On Windows, the Python executable is often simply `python`. Ensure your system's PATH variable is correctly configured, or use `py` if available (e.g., `py -3.13 train_pipeline.py`).
*   **`ModuleNotFoundError`**: This indicates that required Python packages are not installed. Run `pip install -r requirements.txt`.
*   **`InconsistentVersionWarning`**: This warning from `scikit-learn` occurs if models are loaded from a different version than they were trained on. Running `python train_pipeline.py` on your local machine will resolve this by retraining models with your specific `scikit-learn` version.
*   **Models not found**: Ensure you have successfully run `python train_pipeline.py` to generate the model files in the `models/` directory.
*   **Datasets not found**: Verify that `UpdatedResumeDataSet.csv` and `naukri_com-job_sample.csv` are present in `data/raw/`. If not, place them there.
*   **Streamlit port already in use**: If port 8501 is occupied, you can specify an alternative port:
    ```bash
    streamlit run app/streamlit_app.py --server.port 8502
    ```

For further details on project features, structure, and deployment, please refer to `README.md` and `DEPLOYMENT_GUIDE.md`.
