import streamlit as st
import io
import re
import json
import pypdf
import pandas as pd
import traceback
from transformers import pipeline
from huggingface_hub import login

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Professional CV Optimizer",
    page_icon="🚀",
    layout="wide"
)

# --- 2. Self-Contained AI Model Loading ---
@st.cache_resource
def load_model():
    """Loads the Flan-T5-Base model directly into the app."""
    try:
        if 'huggingface' in st.secrets and 'api_token' in st.secrets.huggingface:
            login(token=st.secrets.huggingface.api_token)
        # Increased max_new_tokens to prevent truncation of the JSON output
        pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=2048)
        return pipe
    except Exception as e:
        st.error(f"Fatal Error: Could not load the AI model. Details: {e}")
        st.error(traceback.format_exc())
        return None

pipe = load_model()

# --- 3. Robust AI and Helper Functions ---

def run_local_llm(prompt: str) -> dict:
    """
    Runs a prompt through the local pipeline and uses a more robust method to parse JSON.
    """
    if not pipe:
        return {"error": "The AI model is not loaded."}
    try:
        generated_text = pipe(prompt)[0]['generated_text']
        # A more robust way to find JSON: look for the first '{' and the last '}'
        start = generated_text.find('{')
        end = generated_text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = generated_text[start:end]
            return json.loads(json_str)
        else:
            return {"error": "AI did not return a valid JSON object.", "response": generated_text}
    except Exception as e:
        return {"error": f"An unexpected error occurred during AI processing: {e}"}

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file))
        return "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    except Exception:
        return ""

# --- "Chain of Thought" Functions ---

def get_skills_analysis(resume_text: str, jd_text: str) -> dict:
    """AI Task 1: Extract hard and soft skills."""
    prompt = f"""
Analyze the resume and job description to identify the top 10 hard skills and top 5 soft skills mentioned in the job description. For each skill, count its occurrences in both texts. Return a single, valid JSON object with no other text.
The JSON object must have this structure: {{ "hard_skills": [{{"skill": "Skill Name", "resume_count": integer, "jd_count": integer}}], "soft_skills": [{{"skill": "Skill Name", "resume_count": integer, "jd_count": integer}}] }}

Resume Text: --- {resume_text} ---
Job Description Text: --- {jd_text} ---
"""
    return run_local_llm(prompt)

def get_ats_analysis(resume_text: str, jd_text: str) -> dict:
    """AI Task 2: Analyze resume quality metrics."""
    prompt = f"""
Analyze the resume for the following quality metrics against the job description. Return a single, valid JSON object with no other text.
The JSON object must have this structure: {{ "job_title_match": {{ "is_match": boolean, "jd_job_title": "Job Title from JD", "feedback": "Brief analysis." }}, "measurable_results": {{ "count": integer, "feedback": "Analysis of quantifiable achievements." }}, "resume_tone": {{ "has_negative_words": boolean, "feedback": "Analysis of the resume's tone." }} }}

Resume Text: --- {resume_text} ---
Job Description Text: --- {jd_text} ---
"""
    return run_local_llm(prompt)

def get_recruiter_feedback(resume_text: str, jd_text: str) -> dict:
    """AI Task 3: Provide persona-driven, actionable feedback."""
    prompt = f"""
You are a senior recruiter with 15+ years of experience. Analyze the CV against the job description and provide a professional review. Return a single, valid JSON object with no other text.
The JSON object must have this structure: {{ "match_score": integer, "summary_feedback": "A short, human-like summary of your initial thoughts.", "actionable_edits": [{{ "area_to_improve": "The specific section of the CV.", "suggested_wording": "The exact text to add or replace. Use Markdown for bolding keywords with **double asterisks**.", "reason": "A concise explanation of why this change is crucial." }}] }}
Instructions:
1. Provide a match score from 0-100.
2. If the score is below 90, provide at least 3 high-impact actionable edits.
3. Write in clear, simple English.

Current CV: --- {resume_text} ---
Target Job Description: --- {jd_text} ---
"""
    return run_local_llm(prompt)

# --- Streamlit User Interface ---
st.title("🚀 Professional CV Optimizer")
st.markdown("Get a complete, professional analysis of your resume, including detailed skill breakdowns and personalized feedback from a senior AI recruiter.")
st.divider()

if not pipe:
    st.error("The application could not start because the AI model failed to load. Please check the deployment logs.")
else:
    col1, col2 = st.columns(2)
    with col1: st.subheader("1. Upload Your Resume")
    resume_file = st.file_uploader("Upload resume (PDF)", type=["pdf"], label_visibility="collapsed")
    with col2: st.subheader("2. Paste the Job Description")
    job_description = st.text_area("Paste job description", height=300, label_visibility="collapsed")

    if st.button("✨ Generate Full Analysis Report", type="primary", use_container_width=True):
        if not resume_file or not job_description.strip():
            st.warning("Please upload a resume and paste a job description.")
        else:
            resume_text = extract_text_from_pdf(resume_file.getvalue())
            if resume_text:
                final_analysis = {}
                analysis_failed = False

                # --- Execute the Chain of Tasks ---
                with st.spinner("Step 1/3: Analyzing skills..."):
                    skills_result = get_skills_analysis(resume_text, job_description)
                    if "error" in skills_result:
                        st.error(f"Skill Analysis Failed: {skills_result['error']}")
                        analysis_failed = True
                    else:
                        final_analysis.update(skills_result)

                if not analysis_failed:
                    with st.spinner("Step 2/3: Checking resume quality (ATS)..."):
                        ats_result = get_ats_analysis(resume_text, job_description)
                        if "error" in ats_result:
                            st.error(f"ATS Analysis Failed: {ats_result['error']}")
                            analysis_failed = True
                        else:
                            final_analysis.update(ats_result)

                if not analysis_failed:
                    with st.spinner("Step 3/3: Generating senior recruiter feedback..."):
                        recruiter_result = get_recruiter_feedback(resume_text, job_description)
                        if "error" in recruiter_result:
                            st.error(f"Recruiter Feedback Failed: {recruiter_result['error']}")
                            analysis_failed = True
                        else:
                            final_analysis.update(recruiter_result)
                
                if not analysis_failed:
                    st.success("Analysis Complete! See your comprehensive report below.")
                    
                    # --- Display the Combined Report ---
                    recruiter_tab, search_tab, hard_skills_tab, soft_skills_tab = st.tabs([
                        "🧑‍💼 Senior Recruiter Review", " ATS / Searchability", "Hard Skills Analysis", "Soft Skills Analysis"
                    ])
                    with recruiter_tab:
                        score = final_analysis.get("match_score", 0)
                        st.metric("Overall Match Score", f"{score}/100")
                        st.progress(score)
                        st.info(f"**Recruiter's Summary:**\n\n_{final_analysis.get('summary_feedback', 'N/A')}_")
                        st.divider()
                        if final_analysis.get("actionable_edits"):
                            st.subheader("Actionable Edits to Boost Your Score")
                            for i, edit in enumerate(final_analysis["actionable_edits"], 1):
                                st.markdown(f"**{i}. Improvement Area:** {edit.get('area_to_improve', 'N/A')}")
                                st.markdown("**Suggested Wording:**")
                                st.markdown(f"> {edit.get('suggested_wording', 'N/A')}", unsafe_allow_html=True)
                                st.markdown(f"**Why it Matters:** *{edit.get('reason', 'N/A')}*")
                                st.markdown("---")
                    
                    with search_tab:
                        st.header("Searchability and ATS Compliance")
                        word_count = len(resume_text.split())
                        if 400 <= word_count <= 800: st.success(f"✅ **Word Count:** Your resume has {word_count} words (ideal range).")
                        else: st.warning(f"⚠️ **Word Count:** Your resume has {word_count} words (suggested range is 400-800).")
                        
                        if final_analysis.get('job_title_match'):
                            if final_analysis['job_title_match']['is_match']: st.success(f"✅ **Job Title Match:** {final_analysis['job_title_match']['feedback']}")
                            else: st.error(f"❌ **Job Title Match:** {final_analysis['job_title_match']['feedback']}")
                        
                        if final_analysis.get('measurable_results'):
                             if final_analysis['measurable_results']['count'] >= 3: st.success(f"✅ **Measurable Results:** {final_analysis['measurable_results']['feedback']}")
                             else: st.warning(f"⚠️ **Measurable Results:** {final_analysis['measurable_results']['feedback']}")

                    with hard_skills_tab:
                        st.header("Hard Skills Comparison")
                        if final_analysis.get('hard_skills'):
                            df = pd.DataFrame(final_analysis['hard_skills'])
                            df['Match'] = df['resume_count'].apply(lambda x: "✅" if x > 0 else "❌")
                            st.dataframe(df[['skill', 'Match', 'jd_count']].rename(columns={'skill': 'Skill', 'jd_count': 'Job Description Count'}), use_container_width=True, hide_index=True)

                    with soft_skills_tab:
                        st.header("Soft Skills Comparison")
                        if final_analysis.get('soft_skills'):
                            df = pd.DataFrame(final_analysis['soft_skills'])
                            df['Match'] = df['resume_count'].apply(lambda x: "✅" if x > 0 else "❌")
                            st.dataframe(df[['skill', 'Match', 'jd_count']].rename(columns={'skill': 'Skill', 'jd_count': 'Job Description Count'}), use_container_width=True, hide_index=True)