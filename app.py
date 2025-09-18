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

def run_inference_api(prompt: str, max_tokens: int = 2048) -> dict:
    """Generic function to call the Hugging Face Inference API."""
    try:
        hf_token = st.secrets["huggingface"]["api_token"]
    except (KeyError, FileNotFoundError):
        return {"error": "Hugging Face API token not found in Streamlit Secrets."}

    # --- FINAL MODEL FIX ---
    # Switched to Zephyr-7B-Beta, a powerful and reliably available model.
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "temperature": 0.1, "return_full_text": False}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        generated_text = result[0]['generated_text']
        
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

def get_detailed_analysis(resume_text: str, jd_text: str) -> dict:
    """
    Performs the quantitative analysis for skills, ATS checks, etc.
    """
    # Re-added the instruction format for better model performance.
    prompt = f"""<|system|>
You are an expert AI resume analyst. Your task is to perform a detailed analysis of the provided resume against the job description. Return a single, valid JSON object.</s>
<|user|>
The JSON object must have this structure:
{{
  "hard_skills": [{{"skill": "Skill Name", "resume_count": integer, "jd_count": integer}}],
  "soft_skills": [{{"skill": "Skill Name", "resume_count": integer, "jd_count": integer}}],
  "job_title_match": {{ "is_match": boolean, "jd_job_title": "Job Title from JD", "feedback": "Brief analysis." }},
  "measurable_results": {{ "count": integer, "feedback": "Analysis of quantifiable achievements." }},
  "resume_tone": {{ "has_negative_words": boolean, "feedback": "Analysis of the resume's tone." }}
}}
Instructions:
1.  For "hard_skills" and "soft_skills": Identify the top 10-15 most important skills from the job description. For each skill, count its occurrences in both the resume and the job description.
2.  Analyze job title match, measurable results, and resume tone as described in the JSON structure.

Resume Text: --- {resume_text} ---
Job Description Text: --- {jd_text} ---</s>
<|assistant|>
"""
    return run_inference_api(prompt)

def get_recruiter_feedback(resume_text: str, jd_text: str) -> dict:
    """
    Performs the qualitative analysis using the Senior Recruiter persona.
    """
    prompt = f"""<|system|>
You’re a senior Frontend/Fullstack recruiter & resume writer with 15+ years of experience. Given the user's current CV and a target job description, please perform a detailed analysis. Your tone is encouraging, expert, and direct. Return a single, valid JSON object with no other text or explanation.</s>
<|user|>
The JSON object must have this structure:
{{
  "match_score": integer,
  "summary_feedback": "A short, human-like summary of your initial thoughts.",
  "actionable_edits": [
    {{
      "area_to_improve": "The specific section of the CV (e.g., 'Professional Summary').",
      "suggested_wording": "The exact, human-written text to add or replace. Use Markdown for bolding important keywords with **double asterisks**.",
      "reason": "A concise explanation of why this change is crucial for recruiters and ATS."
    }}
  ]
}}
Instructions:
1. Analyze both and return an initial match score between 0 and 100.
2. Provide a brief `summary_feedback`.
3. If the score is below 90%, the `actionable_edits` array must contain at least 3-5 precise, high-impact edits.
4. Write in clear, simple English. Avoid robotic or overly complex words.

Current CV: --- {resume_text} ---
Target Job Description: --- {jd_text} ---</s>
<|assistant|>
"""
    return run_inference_api(prompt, max_tokens=2048)


# --- Streamlit User Interface ---
st.title("🚀 Professional CV Optimizer")
st.markdown("Get a complete, professional analysis of your resume, including detailed skill breakdowns and personalized feedback from a senior AI recruiter.")
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
        with st.spinner("Performing dual-layer AI analysis... This may take up to 90 seconds."):
            resume_text = extract_text_from_pdf(resume_file.getvalue())
            if resume_text:
                # --- Run BOTH analyses ---
                detailed_analysis = get_detailed_analysis(resume_text, job_description)
                recruiter_feedback = get_recruiter_feedback(resume_text, job_description)

                # --- Error Checking ---
                if "error" in detailed_analysis or "error" in recruiter_feedback:
                    if "error" in detailed_analysis:
                        st.error(f"Detailed Analysis Failed: {detailed_analysis['error']}")
                    if "error" in recruiter_feedback:
                        st.error(f"Recruiter Feedback Failed: {recruiter_feedback['error']}")
                else:
                    st.success("Analysis Complete! See your comprehensive report below.")
                    
                    # --- Create the INTEGRATED Tabbed Report ---
                    recruiter_tab, search_tab, hard_skills_tab, soft_skills_tab = st.tabs([
                        "🧑‍💼 Senior Recruiter Review", " ATS / Searchability", "Hard Skills Analysis", "Soft Skills Analysis"
                    ])

                    with recruiter_tab:
                        st.header("Personalized Recruiter Feedback")
                        score = recruiter_feedback.get("match_score", 0)
                        st.metric("Overall Match Score", f"{score}/100")
                        st.progress(score)
                        
                        st.info(f"**Recruiter's Summary:**\n\n_{recruiter_feedback.get('summary_feedback', 'No summary provided.')}_")
                        st.divider()
                        
                        if recruiter_feedback.get("actionable_edits"):
                            st.subheader("Actionable Edits to Boost Your Score")
                            for i, edit in enumerate(recruiter_feedback["actionable_edits"], 1):
                                st.markdown(f"**{i}. Improvement Area:** {edit.get('area_to_improve', 'General Suggestion')}")
                                st.markdown("**Suggested Wording:**")
                                st.markdown(f"> {edit.get('suggested_wording', 'N/A')}", unsafe_allow_html=True)
                                st.markdown(f"**Why it Matters:** *{edit.get('reason', 'N/A')}*")
                                st.markdown("---")
                        else:
                            st.balloons()
                            st.success("Your CV is highly aligned with this role!")
                            
                    with search_tab:
                        st.header("Searchability and ATS Compliance")
                        # Rule-based checks
                        word_count = len(resume_text.split())
                        if 400 <= word_count <= 800:
                            st.success(f"✅ **Word Count:** Your resume has {word_count} words (in the ideal range).", icon="✅")
                        else:
                             st.warning(f"⚠️ **Word Count:** Your resume has {word_count} words. The suggested range is 400-800.", icon="⚠️")
                        
                        # AI-based checks from the detailed analysis
                        if detailed_analysis.get('job_title_match'):
                            if detailed_analysis['job_title_match']['is_match']:
                                st.success(f"✅ **Job Title Match:** {detailed_analysis['job_title_match']['feedback']}", icon="✅")
                            else:
                                st.error(f"❌ **Job Title Match:** {detailed_analysis['job_title_match']['feedback']}", icon="❌")
                        
                        if detailed_analysis.get('measurable_results'):
                             if detailed_analysis['measurable_results']['count'] >= 3:
                                st.success(f"✅ **Measurable Results:** {detailed_analysis['measurable_results']['feedback']}", icon="✅")
                             else:
                                st.warning(f"⚠️ **Measurable Results:** {detailed_analysis['measurable_results']['feedback']}", icon="⚠️")

                    with hard_skills_tab:
                        st.header("Hard Skills Comparison")
                        if detailed_analysis.get('hard_skills'):
                            df_hard = pd.DataFrame(detailed_analysis['hard_skills'])
                            df_hard.columns = ["Skill", "Resume Count", "Job Description Count"]
                            df_hard['Match'] = df_hard['Resume Count'].apply(lambda x: "✅" if x > 0 else "❌")
                            st.dataframe(df_hard[['Skill', 'Match', 'Job Description Count']], use_container_width=True, hide_index=True)

                    with soft_skills_tab:
                        st.header("Soft Skills Comparison")
                        if detailed_analysis.get('soft_skills'):
                            df_soft = pd.DataFrame(detailed_analysis['soft_skills'])
                            df_soft.columns = ["Skill", "Resume Count", "Job Description Count"]
                            df_soft['Match'] = df_soft['Match'] = df_soft['Resume Count'].apply(lambda x: "✅" if x > 0 else "❌")
                            st.dataframe(df_soft[['Skill', 'Match', 'Job Description Count']], use_container_width=True, hide_index=True)