# SmartHire: Deployment Guide

This guide provides instructions for deploying the SmartHire application to various hosting platforms. The project is designed to be self-contained and can be deployed independently without requiring any external AI platform services.

## 1. Prerequisites

Before deploying, ensure you have:

*   A GitHub account and a repository containing the SmartHire project.
*   An account with a chosen hosting platform (e.g., Streamlit Community Cloud, Render, Hugging Face Spaces).

## 2. General Deployment Steps

Most cloud platforms follow a similar deployment workflow:

1.  **Push your code to GitHub**: Ensure your entire SmartHire project, including all `src/` files, `app/streamlit_app.py`, `requirements.txt`, and trained models (if pre-trained), is pushed to a GitHub repository.
2.  **Create a new application on your chosen platform**: Navigate to your platform's dashboard and initiate the creation of a new web application.
3.  **Connect to your GitHub repository**: Authorize the platform to access your GitHub repository and select the SmartHire project.
4.  **Configure build settings**: Specify the Python version (e.g., Python 3.9, 3.10, 3.11), the build command (usually `pip install -r requirements.txt`), and the run command (e.g., `streamlit run app/streamlit_app.py`).
5.  **Set environment variables (if any)**: For SmartHire, there are no specific environment variables required for basic functionality, but some platforms might require configuration for resource limits.
6.  **Deploy the application**: Start the deployment process. The platform will clone your repository, install dependencies, and run your Streamlit application.
7.  **Access the public URL**: Once deployed, the platform will provide a public HTTPS URL where your SmartHire application can be accessed.

## 3. Platform-Specific Instructions

### 3.1. Streamlit Community Cloud

Streamlit Community Cloud is the easiest way to deploy Streamlit apps.

1.  **Sign up/Log in**: Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2.  **Deploy an app**: Click on "New app" from your dashboard.
3.  **Connect repository**: Select your SmartHire GitHub repository, the branch (e.g., `main`), and the main file path (`app/streamlit_app.py`).
4.  **Advanced settings (optional)**: You can specify Python version or add secrets if needed (not required for basic SmartHire).
5.  **Deploy!**: Click "Deploy!" and wait for your app to build and launch.
6.  **Public URL**: Streamlit will provide a public URL for your application.

### 3.2. Render

Render is a unified cloud to build and run all your apps and websites.

1.  **Sign up/Log in**: Go to [render.com](https://render.com/) and connect your GitHub account.
2.  **New Web Service**: From the dashboard, click "New" -> "Web Service".
3.  **Connect repository**: Select your SmartHire GitHub repository.
4.  **Configure**: 
    *   **Name**: `smarthire-app` (or your preferred name)
    *   **Region**: Choose a region close to your users.
    *   **Branch**: `main` (or your deployment branch)
    *   **Root Directory**: `/`
    *   **Runtime**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `streamlit run app/streamlit_app.py --server.port $PORT --server.enableCORS false --server.enableXsrfProtection false`
        *Note: Render injects a `$PORT` environment variable. Streamlit needs to be configured to use it.*
5.  **Create Web Service**: Click "Create Web Service" and monitor the deployment logs.
6.  **Public URL**: Render will provide a public URL for your application.

### 3.3. Hugging Face Spaces

Hugging Face Spaces provides a simple way to host ML demos.

1.  **Sign up/Log in**: Go to [huggingface.co/spaces](https://huggingface.co/spaces) and log in.
2.  **Create new Space**: Click "Create new Space".
3.  **Configure**: 
    *   **Space name**: `smarthire-app` (or your preferred name)
    *   **License**: Choose an appropriate license.
    *   **Space SDK**: `Streamlit`
    *   **Python version**: Select a compatible Python version.
    *   **Visibility**: Public or Private.
4.  **Create Space**: This will create an empty repository on Hugging Face.
5.  **Push your code**: Clone the Hugging Face Space repository locally, copy your SmartHire project files into it, and push them back to Hugging Face. Ensure `requirements.txt` is in the root and `app/streamlit_app.py` is the main app file.
6.  **Monitor**: The Space will automatically build and deploy your application.
7.  **Public URL**: Hugging Face will provide a public URL for your application.

## 4. Troubleshooting Common Deployment Issues

*   **Dependency Conflicts**: Ensure `requirements.txt` lists all necessary packages and their compatible versions. Consider using `pip freeze > requirements.txt` in your local virtual environment.
*   **Port Binding**: If deploying to platforms like Render, ensure Streamlit is configured to bind to the port provided by the environment variable (e.g., `--server.port $PORT`).
*   **File Paths**: Verify all file paths in your code (especially for data and models) are relative to the project root or handled robustly for the deployment environment.
*   **Resource Limits**: For larger models or datasets, you might need to select a more powerful instance type on your hosting platform.

By following these instructions, you should be able to successfully deploy your SmartHire application.
