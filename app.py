import streamlit as st
import spacy
import pandas as pd
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("spaCy model 'en_core_web_sm' is not installed. Run `python -m spacy download en_core_web_sm` in your terminal.")
    st.stop()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# Function to preprocess text
def preprocess(text):
    if not text:
        return ""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Function to rank resumes
def rank_resumes(job_desc, resumes):
    texts = [job_desc] + resumes
    processed_texts = [preprocess(text) for text in texts]
    tfidf = TfidfVectorizer()
    try:
        tfidf_matrix = tfidf.fit_transform(processed_texts)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        return similarity_scores
    except Exception as e:
        st.error(f"Error in vectorization or similarity computation: {e}")
        return [0.0] * len(resumes)

# Streamlit UI
st.title("🔍 AI-Powered Resume Screening Tool")
st.write("Upload a **Job Description** and multiple **Candidate Resumes in PDF format** to rank the best matches.")

job_desc_input = st.text_area("Job Description", height=200)
resume_files = st.file_uploader("Upload Resumes (PDF only")
