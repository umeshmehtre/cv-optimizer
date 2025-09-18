import streamlit as st
import io
import re
import json
import pypdf
import requests
import pandas as pd
import traceback

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Professional CV Optimizer",
    page_icon="🚀",
    layout="wide"
)

# --- 2. Core AI and Helper Functions ---

def run_inference_api(prompt: str) -> dict:
    """
    Calls a powerful LLM via the Hugging Face Inference API and returns a structured JSON.
    """
    try:
        hf_token = st.secrets["huggingface"]["api_token"]
    except (KeyError, FileNotFoundError):
        return {"error": "Hugging Face API token not found in Streamlit Secrets."}

    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 2048, "temperature": 0.1, "return_full_text": False}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        generated_text = result[0]['generated_text']
        
        # Clean and parse the JSON
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return {"error": "AI did not return a valid JSON object.", "response": generated_text}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"API Request Failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file))
        return "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    except Exception:
        return ""

def perform_full_analysis(resume_text: str, jd_text: str) -> dict:
    """
    Performs a comprehensive analysis using a master AI prompt and rule-based checks.
    """
    # --- Master AI Prompt ---
    # This prompt asks for all the qualitative and quantitative data in one structured call.
    prompt = f"""
[INST]
You are an expert AI career advisor and resume analyst. Your task is to perform a detailed, critical analysis of the provided resume against the job description. Return a single, valid JSON object with no other text or explanation.

The JSON object must have the following exact structure:
{{
  "hard_skills": [{{"skill": "Skill Name", "resume_count": integer, "jd_count": integer}}],
  "soft_skills": [{{"skill": "Skill Name", "resume_count": integer, "jd_count": integer}}],
  "job_title_match": {{ "is_match": boolean, "jd_job_title": "Job Title from JD", "feedback": "Your brief analysis here." }},
  "measurable_results": {{ "count": integer, "feedback": "Your analysis of quantifiable achievements." }},
  "resume_tone": {{ "has_negative_words": boolean, "feedback": "Your analysis of the resume's tone and use of clichés." }}
}}

Instructions:
1.  For "hard_skills" and "soft_skills": Identify the top 10-15 most important skills from the job description. For each skill, count its occurrences in both the resume and the job description.
2.  For "job_title_match": Find the job title in the job description. Check if the resume mentions this exact title. Provide brief feedback.
3.  For "measurable_results": Scan the resume for quantifiable results (e.g., numbers, percentages, dollar amounts). Count how many you find and provide feedback.
4.  For "resume_tone": Analyze the resume for negative phrases or common clichés (e.g., "results-oriented," "team player"). Note if any are found and provide feedback.

Resume Text:
---
{resume_text}
---

Job Description Text:
---
{jd_text}
---
[/INST]
"""
    
    # Get AI analysis
    analysis_results = run_inference_api(prompt)

    # --- Rule-Based Checks ---
    # These are faster and more reliable for simple checks.
    analysis_results['contact_info'] = {
        "email_found": bool(re.search(r'[\w\.-]+@[\w\.-]+', resume_text)),
        "phone_found": bool(re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', resume_text))
    }
    analysis_results['section_headings'] = {
        "education_found": bool(re.search(r'\b(education)\b', resume_text, re.IGNORECASE)),
        "experience_found": bool(re.search(r'\b(experience|work history)\b', resume_text, re.IGNORECASE))
    }
    analysis_results['word_count'] = len(resume_text.split())

    return analysis_results


# --- 5. Streamlit User Interface ---
st.title("🚀 Professional CV Optimizer")
st.markdown("Get a detailed, professional analysis of your resume against a job description, complete with actionable recruiter tips.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Upload Your Resume")
    resume_file = st.file_uploader("Upload resume (PDF)", type=["pdf"], label_visibility="collapsed")
with col2:
    st.subheader("2. Paste the Job Description")
    job_description = st.text_area("Paste job description", height=300, label_visibility="collapsed")

if st.button("✨ Generate Full Analysis Report", type="primary", use_container_width=True):
    if not resume_file or not job_description.strip():
        st.warning("Please upload a resume and paste a job description.")
    else:
        with st.spinner("Performing deep analysis... This is advanced AI and may take up to 60 seconds."):
            resume_text = extract_text_from_pdf(resume_file.getvalue())
            if resume_text:
                analysis = perform_full_analysis(resume_text, job_description)

                if "error" in analysis:
                    st.error(f"Analysis failed: {analysis['error']}")
                    if "response" in analysis: st.code(analysis['response'])
                else:
                    st.success("Analysis Complete! See your detailed report below.")
                    
                    # --- Create the Tabbed Report ---
                    search_tab, hard_skills_tab, soft_skills_tab, recruiter_tab = st.tabs([
                        " ATS / Searchability", "Hard Skills Analysis", "Soft Skills Analysis", "Recruiter Tips"
                    ])

                    with search_tab:
                        st.header("Searchability and ATS Compliance")
                        st.markdown("How well your resume is parsed by automated systems and found by recruiters.")
                        
                        # Contact Info
                        if analysis['contact_info']['email_found']:
                            st.success("✅ **Contact Info:** You provided your email.", icon="✅")
                        if analysis['contact_info']['phone_found']:
                            st.success("✅ **Contact Info:** You provided your phone number.", icon="✅")
                        
                        # Section Headings
                        if analysis['section_headings']['education_found']:
                            st.success("✅ **Section Headings:** We found an education section.", icon="✅")
                        if analysis['section_headings']['experience_found']:
                            st.success("✅ **Section Headings:** We found a work experience section.", icon="✅")
                        
                        # Job Title Match
                        if analysis.get('job_title_match'):
                            if analysis['job_title_match']['is_match']:
                                st.success(f"✅ **Job Title Match:** {analysis['job_title_match']['feedback']}", icon="✅")
                            else:
                                st.error(f"❌ **Job Title Match:** {analysis['job_title_match']['feedback']}", icon="❌")

                    with hard_skills_tab:
                        st.header("Hard Skills Comparison")
                        st.markdown("**Tip:** Match the skills in your resume to the exact spelling in the job description. Prioritize skills that appear most frequently.")
                        if analysis.get('hard_skills'):
                            df_hard = pd.DataFrame(analysis['hard_skills'])
                            df_hard.columns = ["Skill", "Resume Count", "Job Description Count"]
                            df_hard['Resume Count'] = df_hard['Resume Count'].apply(lambda x: "✅" if x > 0 else "❌")
                            st.dataframe(df_hard, use_container_width=True, hide_index=True)

                    with soft_skills_tab:
                        st.header("Soft Skills Comparison")
                        st.markdown("**Tip:** Prioritize hard skills in your resume to get interviews, and then showcase your soft skills in the interview itself.")
                        if analysis.get('soft_skills'):
                            df_soft = pd.DataFrame(analysis['soft_skills'])
                            df_soft.columns = ["Skill", "Resume Count", "Job Description Count"]
                            df_soft['Resume Count'] = df_soft['Resume Count'].apply(lambda x: "✅" if x > 0 else "❌")
                            st.dataframe(df_soft, use_container_width=True, hide_index=True)
                            
                    with recruiter_tab:
                        st.header("Recruiter Tips")
                        st.markdown("Feedback on the finer points that recruiters notice.")
                        
                        if analysis.get('measurable_results'):
                            if analysis['measurable_results']['count'] >= 3:
                                st.success(f"✅ **Measurable Results:** {analysis['measurable_results']['feedback']}", icon="✅")
                            else:
                                st.warning(f"⚠️ **Measurable Results:** {analysis['measurable_results']['feedback']}", icon="⚠️")
                        
                        if analysis.get('resume_tone'):
                            if not analysis['resume_tone']['has_negative_words']:
                                st.success(f"✅ **Resume Tone:** {analysis['resume_tone']['feedback']}", icon="✅")
                            else:
                                st.warning(f"⚠️ **Resume Tone:** {analysis['resume_tone']['feedback']}", icon="⚠️")
                        
                        # Word Count
                        if 400 <= analysis['word_count'] <= 800:
                            st.success(f"✅ **Word Count:** Your resume has {analysis['word_count']} words, which is in the suggested range.", icon="✅")
                        else:
                             st.warning(f"⚠️ **Word Count:** Your resume has {analysis['word_count']} words. The suggested range is 400-800 words for optimal readability.", icon="⚠️")