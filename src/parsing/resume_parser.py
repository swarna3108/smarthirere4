import os
import re
from pdfminer.high_level import extract_text
import docx

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_txt(txt_path):
    """Extract text from a TXT file."""
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        return ""

def parse_resume(file_path):
    """Extract text from resume based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        return ""

def clean_resume_text(text):
    """Clean the extracted resume text."""
    # Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
    # Remove RTF tags
    text = re.sub(r'RTF', ' ', text)
    # Remove hashtags
    text = re.sub(r'#\S+', ' ', text)
    # Remove mentions
    text = re.sub(r'@\S+', ' ', text)
    # Remove punctuations
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    # Remove non-ascii characters
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()
