import streamlit as st
import io
import re
import json
import pypdf  # Using the modern pypdf library
from transformers import pipeline
from huggingface_hub import login
import traceback

# --- 1. Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="CV Resume Optimizer",
    page_icon="📄",
    layout="wide"
)

# --- 2. AI Model Loading (Cached for performance) ---
@st.cache_resource
def load_model():
    """Load the Hugging Face model. Logs in if a token is available."""
    try:
        # This part is for Streamlit Cloud deployment
        if 'huggingface' in st.secrets and 'token' in st.secrets.huggingface:
            login(token=st.secrets.huggingface.token)
            
        # Using a memory-friendly model suitable for free hosting
        model_name = "t5-small"
        pipeline_instance = pipeline("text2text-generation", model=model_name)
        return pipeline_instance
    except Exception as e:
        st.error(f"Error loading AI model: {e}")
        st.error(traceback.format_exc())
        return None

llm_pipeline = load_model()

# --- 3. Helper Functions (The app's "brain") ---

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def run_llm_prompt(prompt: str, max_length: int = 512) -> str:
    """Runs a prompt through the AI model and gets a response."""
    if not llm_pipeline:
        return "Error: AI model is not loaded."
    try:
        results = llm_pipeline(prompt, max_length=max_length, num_beams=4, early_stopping=True)
        return results[0]['generated_text']
    except Exception as e:
        return f"Error during AI generation: {e}"

def extract_skills_from_job_description(job_description: str) -> list:
    """Uses AI to find and categorize skills in a job description."""
    prompt = f"""
    Analyze the job description below. Extract key skills.
    Categorize each skill as "hard" or "soft" and rate its importance as "high", "medium", or "low".
    Return ONLY a valid JSON object with a key "skills" containing a list of objects.

    Job Description: {job_description}
    """
    response_text = run_llm_prompt(prompt, max_length=1024)
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0)).get('skills', [])
        return []
    except (json.JSONDecodeError, AttributeError):
        return []

# --- 4. Streamlit User Interface (What the user sees) ---

st.title("📄 CV Resume Optimizer")
st.markdown("Upload your resume and paste a job description to get AI-powered feedback and improve your match score.")
st.divider()

# Check if the model loaded successfully before showing the UI
if not llm_pipeline:
    st.error("The AI model failed to load. The application cannot continue. Please check the logs.")
else:
    # Create two columns for the layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Upload Your Resume")
        resume_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"], label_visibility="collapsed")

    with col2:
        st.subheader("2. Paste the Job Description")
        job_description = st.text_area("Paste the full job description here", height=250, label_visibility="collapsed")

    if st.button("Analyze My Resume", type="primary", use_container_width=True):
        if not resume_file:
            st.warning("Please upload your resume.")
        elif not job_description.strip():
            st.warning("Please paste the job description.")
        else:
            with st.spinner("AI is analyzing your documents... Please wait."):
                # --- 5. Analysis Logic (When the button is clicked) ---
                resume_bytes = resume_file.getvalue()
                resume_text = extract_text_from_pdf(resume_bytes)

                if resume_text:
                    required_skills = extract_skills_from_job_description(job_description)
                    # (Further analysis logic for skill matching, etc., would go here)
                    
                    st.success("Analysis Complete!")
                    st.subheader("Extracted Skills from Job Description:")
                    st.json(required_skills)
                    
                    st.subheader("Extracted Text from Your Resume:")
                    st.text_area("Resume Content", resume_text, height=300)