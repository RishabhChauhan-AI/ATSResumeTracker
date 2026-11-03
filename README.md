# ATS Resume Tracker

An AI-powered ATS Resume Tracker built with **Python, Streamlit, and Google's Gemini API**. The application analyzes resumes, evaluates their strengths and weaknesses, and compares them against job descriptions to help candidates improve their chances of passing Applicant Tracking Systems (ATS).

## Features

### Resume Analysis
- Upload and analyze PDF resumes.
- Generate a detailed resume assessment.
- Identify strengths and weaknesses.
- Review technical skills and experience.
- Highlight missing skills and areas for improvement.
- Get recommendations for improving your resume.
- Receive relevant project suggestions and interview preparation tips.

### Job Description Matching
- Compare a resume against a job description.
- Calculate an overall job match score.
- Evaluate skills, experience, education, and keywords.
- Identify matching and missing skills.
- Get recommendations to improve job compatibility.
- Receive targeted interview preparation suggestions.

## Tech Stack

- **Python** – Application logic
- **Streamlit** – Web application interface
- **Google Gemini API** – AI-powered resume analysis
- **PyPDF** – PDF text extraction
- **Python-dotenv** – Environment variable management

## Project Structure

```text
ATSResumeTracker/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ATSResumeTracker.git
```

### 2. Navigate to the project directory

```bash
cd ATSResumeTracker
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure your API key

Create a `.env` file in the project root directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

Replace the placeholder with your own Google Gemini API key.

### 7. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## How It Works

1. Upload your resume in PDF format.
2. Choose between Resume Analysis and Job Description Matching.
3. For job matching, enter the job description.
4. The application extracts text from the uploaded PDF.
5. Gemini analyzes the resume and generates a structured report.
6. Review your match score, skills, weaknesses, and improvement recommendations.

## Security

- API keys are stored in environment variables.
- Never upload your `.env` file to GitHub.
- Keep your API credentials private.

## Future Improvements

- Resume comparison across multiple job descriptions.
- Support for additional document formats.
- Resume keyword optimization.
- Improved ATS scoring and evaluation.
- Resume improvement suggestions based on specific job roles.

## Author

**YOUR NAME**

GitHub: [MY_GITHUB_PROFILE](https://github.com/RishabhChauhan-AI)

---

*Built as a practical AI-powered project to explore resume analysis, ATS evaluation, and Generative AI integration.*