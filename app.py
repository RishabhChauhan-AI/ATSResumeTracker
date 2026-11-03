
import os
import io

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="ATS Resume Tracker",
    page_icon=":material/description:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 750;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .sub-title {
        color: #888888;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        padding: 16px;
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 9px;
        min-height: 45px;
        font-weight: 600;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 2. HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">ATS Resume Tracker</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Analyze your resume, evaluate job compatibility, '
    'and discover opportunities to improve.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# 3. API CONFIGURATION
# --------------------------------------------------

api_key = os.getenv("GOOGLE_API_KEY")

try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

if not api_key:
    st.error("Application configuration is missing.")
    st.stop()


@st.cache_resource
def load_client(key):
    return genai.Client(api_key=key)


client = load_client(api_key)


# Preferred model order
PREFERRED_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash"
]


# --------------------------------------------------
# 4. MODEL AVAILABILITY
# --------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_available_models():

    available_models = []

    try:
        for model in client.models.list():

            name = model.name.replace("models/", "")

            supported_actions = getattr(
                model,
                "supported_actions",
                []
            ) or []

            if (
                name in PREFERRED_MODELS
                and (
                    not supported_actions
                    or "generateContent" in supported_actions
                )
            ):
                available_models.append(name)

    except Exception:
        return PREFERRED_MODELS

    sorted_models = []

    for preferred in PREFERRED_MODELS:
        if preferred in available_models:
            sorted_models.append(preferred)

    return sorted_models if sorted_models else PREFERRED_MODELS


# --------------------------------------------------
# 5. PDF TEXT EXTRACTION
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def extract_resume_text(pdf_bytes):

    reader = PdfReader(io.BytesIO(pdf_bytes))

    if reader.is_encrypted:
        raise ValueError(
            "Password-protected PDFs are not supported."
        )

    resume_text = ""

    for page in reader.pages:

        page_text = page.extract_text() or ""

        resume_text += page_text + "\n"

        if len(resume_text) >= 30000:
            break

    resume_text = resume_text.strip()

    if not resume_text:
        raise ValueError(
            "No readable text was found. "
            "Your PDF may be scanned or image-based."
        )

    return resume_text[:30000]


# --------------------------------------------------
# 6. ANALYSIS PROMPTS
# --------------------------------------------------

def get_analysis_prompt(analysis_type):

    common_rules = """
    IMPORTANT RULES:

    1. Use only evidence present in the resume and job description.
    2. Never invent experience, skills, qualifications or achievements.
    3. If information is missing, say that it is not demonstrated.
    4. Distinguish between a missing skill and missing evidence.
    5. Be specific. Avoid generic feedback.
    6. Give practical recommendations.
    7. Do not assume that the candidate is unsuitable because
       they are a fresher.
    8. Return valid JSON only. Do not use Markdown code fences.
    """

    if analysis_type == "Resume Review":

        prompt = """
        You are an experienced technical recruiter and resume specialist.

        Review the candidate's resume in relation to the supplied
        job description.

        Return a JSON object with exactly these fields:

        {
            "candidate_overview": "Detailed candidate summary",
            "overall_assessment": "Overall professional assessment",
            "strengths": [
                {
                    "title": "Strength",
                    "explanation": "Detailed explanation",
                    "evidence": "Evidence from resume"
                }
            ],
            "technical_skills": [
                {
                    "skill": "Skill name",
                    "status": "Strong or Relevant or Needs Improvement",
                    "explanation": "Reason"
                }
            ],
            "experience_and_projects": [
                {
                    "title": "Project or experience",
                    "relevance": "High or Medium or Low",
                    "explanation": "How it relates to the job",
                    "improvement": "How to present it better"
                }
            ],
            "weaknesses": [
                {
                    "title": "Weakness",
                    "explanation": "Detailed explanation",
                    "recommendation": "How to improve"
                }
            ],
            "missing_skills": [
                {
                    "skill": "Missing or unproven skill",
                    "importance": "High or Medium or Low",
                    "reason": "Why it matters"
                }
            ],
            "resume_improvements": [
                {
                    "area": "Resume section",
                    "current_issue": "Current issue",
                    "suggestion": "Specific improvement",
                    "example": "Suggested example wording"
                }
            ],
            "project_recommendations": [
                {
                    "project": "Suggested project",
                    "skills": "Skills it demonstrates",
                    "reason": "Why it would help"
                }
            ],
            "interview_preparation": [
                "Potential interview preparation topic"
            ],
            "final_recommendations": [
                "Actionable recommendation"
            ]
        }

        Provide detailed, role-specific feedback.
        Include as many relevant points as the evidence supports.
        Do not create fictional information.
        """

    else:

        prompt = """
        You are an experienced technical recruiter and resume
        evaluation specialist.

        Compare the resume with the job description.

        Evaluate these five criteria:

        1. Technical Skills: 35
        2. Experience and Projects: 25
        3. Education and Qualifications: 15
        4. Keywords and Tools: 15
        5. Other Job Requirements: 10

        For every criterion, assign a score from 0 to 100.
        The application will calculate the weighted final score.

        Return a JSON object with exactly these fields:

        {
            "candidate_overview": "Detailed candidate summary",
            "overall_assessment": "Detailed overall assessment",
            "criteria": [
                {
                    "criterion": "Technical Skills",
                    "weight": 35,
                    "score": 0,
                    "evidence": "Specific evidence from resume",
                    "gaps": "What is missing or not demonstrated"
                },
                {
                    "criterion": "Experience and Projects",
                    "weight": 25,
                    "score": 0,
                    "evidence": "Specific evidence from resume",
                    "gaps": "What is missing or not demonstrated"
                },
                {
                    "criterion": "Education and Qualifications",
                    "weight": 15,
                    "score": 0,
                    "evidence": "Specific evidence from resume",
                    "gaps": "What is missing or not demonstrated"
                },
                {
                    "criterion": "Keywords and Tools",
                    "weight": 15,
                    "score": 0,
                    "evidence": "Specific evidence from resume",
                    "gaps": "What is missing or not demonstrated"
                },
                {
                    "criterion": "Other Job Requirements",
                    "weight": 10,
                    "score": 0,
                    "evidence": "Specific evidence from resume",
                    "gaps": "What is missing or not demonstrated"
                }
            ],
            "matching_skills": [
                {
                    "skill": "Skill name",
                    "evidence": "Where it appears in resume"
                }
            ],
            "missing_skills": [
                {
                    "skill": "Skill name",
                    "importance": "High or Medium or Low",
                    "reason": "Why it matters"
                }
            ],
            "strengths": [
                {
                    "title": "Strength",
                    "explanation": "Detailed explanation"
                }
            ],
            "improvement_areas": [
                {
                    "title": "Improvement area",
                    "explanation": "Detailed explanation",
                    "recommendation": "Practical action"
                }
            ],
            "project_recommendations": [
                {
                    "project": "Suggested project",
                    "skills": "Skills demonstrated",
                    "reason": "Why it improves the candidate's fit"
                }
            ],
            "interview_preparation": [
                "Potential interview question or topic"
            ],
            "final_recommendations": [
                "Actionable recommendation"
            ]
        }

        Make the scores realistic and evidence-based.
        Do not give a high score merely because a keyword appears once.
        Do not treat an unmentioned skill as proof that the candidate
        cannot perform it.
        """

    return prompt + "\n\n" + common_rules


# --------------------------------------------------
# 7. MODEL ANALYSIS
# --------------------------------------------------

def get_analysis(job_description, resume_text, analysis_type):

    models = get_available_models()

    prompt = get_analysis_prompt(analysis_type)

    contents = [
        prompt,
        "RESUME:\n" + resume_text,
        "JOB DESCRIPTION:\n" + job_description
    ]

    last_error = None

    for model_name in models[:2]:

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=5000,
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(
                        timeout=45000,
                        retry_options=types.HttpRetryOptions(
                            attempts=1
                        )
                    )
                )
            )

            if not response.text:
                raise ValueError(
                    "The server returned an empty response."
                )

            import json

            return json.loads(response.text)

        except Exception as e:

            last_error = e
            error = str(e).lower()

            retryable = (
                "503" in error
                or "429" in error
                or "500" in error
                or "404" in error
                or "not found" in error
                or "unavailable" in error
                or "overloaded" in error
                or "timeout" in error
                or "timed out" in error
                or "resource_exhausted" in error
                or "not supported" in error
            )

            if retryable:
                continue

            raise

    raise Exception(
        "The server is currently unavailable. "
        "Please try again later."
    ) from last_error


# --------------------------------------------------
# 8. MATCH SCORE CALCULATION
# --------------------------------------------------

def calculate_match_score(criteria):

    total_score = 0
    total_weight = 0

    for item in criteria:

        weight = float(item.get("weight", 0))
        score = float(item.get("score", 0))

        score = max(0, min(100, score))

        total_score += weight * score
        total_weight += weight

    if total_weight == 0:
        return 0

    return round(total_score / total_weight)


# --------------------------------------------------
# 9. REPORT DISPLAY HELPERS
# --------------------------------------------------

def show_list(items, title_key="title", description_key="explanation"):

    if not items:
        st.info("No information available in this section.")
        return

    for item in items:

        if isinstance(item, dict):

            title = item.get(title_key, "Details")

            description = item.get(
                description_key,
                ""
            )

            with st.container(border=True):

                st.markdown("**" + str(title) + "**")

                if description:
                    st.write(description)

                for key, value in item.items():

                    if key not in [
                        title_key,
                        description_key
                    ]:
                        if value:
                            st.markdown(
                                f"**{key.replace('_', ' ').title()}:** "
                                f"{value}"
                            )

        else:

            st.markdown("- " + str(item))


# --------------------------------------------------
# 10. MATCH REPORT
# --------------------------------------------------

def show_match_report(report):

    criteria = report.get("criteria", [])

    score = calculate_match_score(criteria)

    st.subheader("Job Compatibility")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Estimated Match",
            f"{score}%"
        )

    with col2:
        st.metric(
            "Matching Skills",
            len(report.get("matching_skills", []))
        )

    with col3:
        st.metric(
            "Missing Skills",
            len(report.get("missing_skills", []))
        )

    st.progress(score / 100)

    st.caption(
        "This is an AI-generated estimate, not an actual employer ATS score."
    )

    st.divider()

    st.subheader("Candidate Overview")

    st.write(report.get("candidate_overview", ""))

    st.subheader("Weighted Match Breakdown")

    for item in criteria:

        criterion = item.get("criterion", "Criterion")
        weight = item.get("weight", 0)
        criterion_score = item.get("score", 0)

        st.markdown(
            f"**{criterion}** · Weight: {weight}%"
        )

        st.progress(
            max(0, min(100, float(criterion_score))) / 100
        )

        st.caption(
            f"Score: {criterion_score}%"
        )

        st.write(item.get("evidence", ""))

        if item.get("gaps"):
            st.caption("Gap: " + str(item["gaps"]))

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Skills",
        "Strengths & Gaps",
        "Projects",
        "Interview Prep",
        "Final Advice"
    ])

    with tab1:

        st.markdown("#### Matching Skills")

        show_list(
            report.get("matching_skills", []),
            "skill",
            "evidence"
        )

        st.markdown("#### Missing Skills")

        show_list(
            report.get("missing_skills", []),
            "skill",
            "reason"
        )

    with tab2:

        st.markdown("#### Strengths")

        show_list(report.get("strengths", []))

        st.markdown("#### Improvement Areas")

        show_list(report.get("improvement_areas", []))

    with tab3:

        st.markdown("#### Recommended Projects")

        show_list(
            report.get("project_recommendations", []),
            "project",
            "reason"
        )

    with tab4:

        st.markdown("#### Topics to Prepare")

        show_list(
            report.get("interview_preparation", [])
        )

    with tab5:

        st.markdown("#### Overall Assessment")

        st.write(report.get("overall_assessment", ""))

        st.markdown("#### Recommended Next Steps")

        show_list(
            report.get("final_recommendations", [])
        )


# --------------------------------------------------
# 11. RESUME REVIEW REPORT
# --------------------------------------------------

def show_review_report(report):

    st.subheader("Resume Review")

    st.markdown("### Candidate Overview")

    st.write(report.get("candidate_overview", ""))

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Strengths",
        "Technical Skills",
        "Experience",
        "Weaknesses",
        "Improvement Plan"
    ])

    with tab1:

        show_list(report.get("strengths", []))

    with tab2:

        show_list(
            report.get("technical_skills", []),
            "skill",
            "explanation"
        )

    with tab3:

        show_list(
            report.get("experience_and_projects", []),
            "title",
            "explanation"
        )

    with tab4:

        st.markdown("#### Weaknesses")

        show_list(report.get("weaknesses", []))

        st.markdown("#### Missing Skills")

        show_list(
            report.get("missing_skills", []),
            "skill",
            "reason"
        )

    with tab5:

        st.markdown("#### Resume Improvements")

        show_list(
            report.get("resume_improvements", []),
            "area",
            "suggestion"
        )

        st.markdown("#### Recommended Projects")

        show_list(
            report.get("project_recommendations", []),
            "project",
            "reason"
        )

        st.markdown("#### Interview Preparation")

        show_list(report.get("interview_preparation", []))

        st.markdown("#### Final Recommendations")

        show_list(report.get("final_recommendations", []))

    st.divider()

    st.markdown("### Overall Assessment")

    st.write(report.get("overall_assessment", ""))


# --------------------------------------------------
# 12. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("ATS Tracker")

    st.caption("Resume analysis workspace")

    st.divider()

    st.markdown("### How it works")

    st.markdown("""
    1. Upload your resume.
    2. Paste a job description.
    3. Choose an analysis.
    4. Review your detailed report.
    """)

    st.divider()

    st.markdown("### Supported files")

    st.write("PDF resumes")

    st.caption("Maximum file size: 10 MB")

    st.divider()

    if st.button(
        "Clear Previous Report",
        use_container_width=True
    ):

        st.session_state.pop("analysis_report", None)
        st.session_state.pop("analysis_type", None)

        st.rerun()


# --------------------------------------------------
# 13. INPUT SECTION
# --------------------------------------------------

st.markdown("## Resume Analysis")

left, right = st.columns([1, 1])

with left:

    st.markdown("### Upload Resume")

    uploaded_file = st.file_uploader(
        "Choose your PDF resume",
        type=["pdf"],
        help="Upload a text-based PDF resume."
    )

    if uploaded_file:

        st.success("Resume uploaded successfully.")

        st.caption(
            f"{uploaded_file.name} · "
            f"{uploaded_file.size / 1024:.1f} KB"
        )

with right:

    st.markdown("### Job Description")

    job_description = st.text_area(
        "Paste the complete job description",
        height=180,
        placeholder=(
            "Paste the job responsibilities, "
            "required skills and qualifications..."
        )
    )


st.divider()

st.markdown("### Choose Analysis")

col1, col2 = st.columns(2)

with col1:

    review_button = st.button(
        "Review My Resume",
        type="secondary",
        use_container_width=True
    )

with col2:

    match_button = st.button(
        "Calculate Job Match",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# 14. VALIDATION AND ANALYSIS
# --------------------------------------------------

if review_button or match_button:

    if uploaded_file is None:

        st.warning("Please upload your resume.")

    elif not job_description.strip():

        st.warning("Please enter a job description.")

    elif uploaded_file.size > 10 * 1024 * 1024:

        st.warning(
            "Your PDF is larger than 10 MB. "
            "Please upload a smaller file."
        )

    elif not uploaded_file.getvalue().startswith(b"%PDF"):

        st.warning("The uploaded file is not a valid PDF.")

    else:

        analysis_type = (
            "Resume Review"
            if review_button
            else "Job Match"
        )

        try:

            with st.spinner("Extracting resume information..."):

                resume_text = extract_resume_text(
                    uploaded_file.getvalue()
                )

            with st.spinner(
                "Analyzing your resume. This may take a moment..."
            ):

                report = get_analysis(
                    job_description,
                    resume_text,
                    analysis_type
                )

            st.session_state["analysis_report"] = report
            st.session_state["analysis_type"] = analysis_type

            st.success("Analysis completed successfully.")

        except Exception as e:

            error = str(e).lower()

            st.error("Unable to complete the analysis.")

            if "timeout" in error or "timed out" in error:

                st.warning(
                    "The server took too long to respond. "
                    "Please try again."
                )

            elif (
                "429" in error
                or "rate limit" in error
                or "resource_exhausted" in error
            ):

                st.warning(
                    "The server is receiving too many requests. "
                    "Please wait before trying again."
                )

            elif (
                "503" in error
                or "unavailable" in error
                or "overloaded" in error
                or "500" in error
            ):

                st.warning(
                    "The server is currently overloaded. "
                    "Please try again later."
                )

            elif "password" in error:

                st.warning(
                    "This PDF is password-protected."
                )

            elif "readable text" in error:

                st.warning(
                    "No readable text was found in the PDF. "
                    "Scanned resumes need OCR."
                )

            else:

                st.warning(
                    "A technical issue occurred. "
                    "Please check the technical details below."
                )

            with st.expander("Technical error details"):

                st.code(str(e))


# --------------------------------------------------
# 15. DISPLAY SAVED REPORT
# --------------------------------------------------

if "analysis_report" in st.session_state:

    report = st.session_state["analysis_report"]

    analysis_type = st.session_state["analysis_type"]

    st.divider()

    st.markdown("## Your Analysis Report")

    if analysis_type == "Job Match":

        show_match_report(report)

    else:

        show_review_report(report)


# --------------------------------------------------
# 16. FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray; padding:10px;">
        <strong>ATS Resume Tracker</strong>
        <br>
        AI-Powered Resume Analysis
        <br>
        <small>
            AI-generated assessments are estimates and should
            not replace human recruitment decisions.
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
