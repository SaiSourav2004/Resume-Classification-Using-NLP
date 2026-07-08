import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import PyPDF2
import docx
from io import StringIO

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="AI Resume Screening & ATS System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Theme Optimized CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 35px;
    }
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1F2937 100%);
        border: 1px solid #374151;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px -1px rgba(59, 130, 246, 0.15);
        text-align: center;
        border-top: 5px solid #3B82F6;
        height: 100%;
    }
    .category-highlight {
        font-size: 2rem;
        font-weight: 800;
        color: #10B981;
        margin-top: 10px;
    }
    .confidence-score {
        font-size: 1rem;
        font-weight: 600;
        color: #9CA3AF;
        margin-top: 5px;
    }
    .ats-excellent { color: #10B981; font-weight: 800; font-size: 2.5rem; margin-top: 5px;}
    .ats-good { color: #F59E0B; font-weight: 800; font-size: 2.5rem; margin-top: 5px;}
    .ats-poor { color: #EF4444; font-weight: 800; font-size: 2.5rem; margin-top: 5px;}
    
    .badge-found {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 5px;
        display: inline-block;
        border: 1px solid #34D399;
    }
    .badge-missing {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 5px;
        display: inline-block;
        border: 1px solid #F87171;
    }
    .suggestion-box {
        background-color: #1F2937;
        border-left: 5px solid #2563EB;
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# RESOURCE INITIALIZATION
# ==========================================
@st.cache_resource
def download_nltk_data():
    """Download required NLTK datasets safely."""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        st.error(f"Error loading NLTK data: {e}")

download_nltk_data()

@st.cache_resource
def load_pipeline():
    """Load the trained ML pipeline containing vectorizer and classifier."""
    try:
        pipeline = joblib.load('resume_pipeline.pkl')
        return pipeline
    except FileNotFoundError:
        return None

# ==========================================
# CORE NLP & ATS FUNCTIONS
# ==========================================
skills_db = {
    "HR": ["recruitment", "onboarding", "employee relations", "payroll", "compliance", "training", "performance management", "sourcing", "screening", "hris"],
    "INFORMATION-TECHNOLOGY": ["python", "java", "sql", "aws", "agile", "networking", "troubleshooting", "linux", "cloud", "security", "database", "api", "docker", "kubernetes"],
    "BUSINESS-DEVELOPMENT": ["sales", "negotiation", "crm", "lead generation", "marketing", "strategy", "presentations", "client relations", "b2b", "cold calling"],
    "FINANCE": ["financial modeling", "accounting", "excel", "forecasting", "auditing", "risk management", "taxation", "reconciliation", "erp"],
    "ENGINEERING": ["autocad", "matlab", "project management", "quality assurance", "manufacturing", "design", "solidworks", "testing", "lean"],
    "HEALTHCARE": ["patient care", "emr", "cpr", "medical terminology", "compliance", "hipaa", "vitals", "nursing", "rehabilitation", "triage"],
    "TEACHER": ["lesson planning", "classroom management", "curriculum development", "special education", "tutoring", "instructional design", "mentoring"],
    "DEFAULT": ["communication", "teamwork", "problem solving", "time management", "leadership", "organization", "project management", "analytical", "critical thinking"]
}

def preprocess_text(text):
    """Text preprocessing[cite: 5]"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    stop_words = set(stopwords.words("english"))
    words = [word for word in text.split() if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def extract_features(raw_text):
    """Extract structural features[cite: 5]"""
    char_count = len(raw_text)
    word_count = len(str(raw_text).split())
    sentence_count = len(re.findall(r"[.!?]", str(raw_text)))
    if sentence_count == 0: sentence_count = 1 
    avg_word_length = sum(len(word) for word in str(raw_text).split()) / max(word_count, 1)
    return char_count, word_count, sentence_count, round(avg_word_length, 2)

def extract_skills(text, category):
    category_key = category.upper()
    expected_skills = skills_db.get(category_key, skills_db["DEFAULT"])
    found_skills = [skill for skill in expected_skills if skill.lower() in text.lower()]
    return found_skills, expected_skills

def calculate_ats_score(found_skills, expected_skills):
    if not expected_skills: 
        return 0
    return int((len(found_skills) / len(expected_skills)) * 100)

def get_missing_skills(found_skills, expected_skills):
    return [skill for skill in expected_skills if skill not in found_skills]

def generate_suggestions(ats_score, missing_skills, word_count, sent_count, avg_word_len):
    suggestions = []
    if word_count < 250:
        suggestions.append("🔸 **Increase Content Length:** This resume is relatively brief. Consider adding more details about past projects, quantifiable achievements, and daily responsibilities.")
    if ats_score < 70 and missing_skills:
        suggestions.append(f"🔸 **Keyword Optimization Required:** The ATS score is below optimal. Naturally integrate missing skills like **{', '.join(missing_skills[:3]).title()}** to improve tracking software visibility.")
    if avg_word_len < 5.5:
        suggestions.append("🔸 **Enhance Vocabulary:** The average word length is slightly low. Opt for more robust, industry-standard professional terminology where appropriate.")
    if sent_count > 0 and (word_count / sent_count) > 30:
        suggestions.append("🔸 **Improve Readability:** Sentences are quite long. Break down long paragraphs into bullet points to improve scannability for human recruiters.")
        
    if not suggestions:
        suggestions.append("⭐ **Outstanding Resume!** The keyword density, length, and structural metrics are perfectly optimized for ATS systems and human readability.")
        
    return suggestions

def extract_text_from_file(uploaded_file):
    """Extracts text based on the file extension."""
    if uploaded_file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.name.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        return " ".join([p.text for p in doc.paragraphs])
    elif uploaded_file.name.endswith('.txt'):
        return str(uploaded_file.read(), "utf-8")
    return ""

# ==========================================
# MAIN APPLICATION UI
# ==========================================

# Sidebar
with st.sidebar:
    st.image("resume_logo.jpeg", width=90)

    st.markdown("## ⚙️ System Configuration")
    st.info("""
    **About this Project:**
    Recruiters often receive hundreds of resumes for a single opening, causing delays. This automated AI system evaluates and shortlists candidates to improve efficiency.
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Status")
    
    pipeline = load_pipeline()
    if pipeline is not None:
        st.success("✅ `resume_pipeline.pkl` loaded.")
    else:
        st.error("⚠️ Pipeline Not Found. Please place `resume_pipeline.pkl` in the root folder.")
        
    st.markdown("---")
    st.markdown("🎓 **Data Science Capstone**")
    st.markdown("Developed using Scikit-Learn, NLTK & Streamlit.")

# Header
st.markdown("<div class='main-header'>AI-Powered Resume Screening System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Instantly classify resumes, extract skills, and generate actionable ATS insights</div>", unsafe_allow_html=True)

# Input Section
st.markdown("### 📝 Upload or Enter Candidate Resume")

uploaded_file = st.file_uploader("Upload Resume File (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

extracted_text = ""
if uploaded_file is not None:
    extracted_text = extract_text_from_file(uploaded_file)

resume_input = st.text_area("Or paste the plain text contents of the candidate's resume below:", 
                            value=extracted_text,
                            height=200, 
                            placeholder="e.g. Highly motivated Information Technology professional with 5 years of experience in Python, AWS, and Agile methodologies...")

if st.button("🚀 Analyze & Predict", use_container_width=True, type="primary"):
    if not resume_input.strip():
        st.warning("⚠️ Please provide resume text to analyze.")
    elif pipeline is None:
        st.error("⚠️ Cannot predict: `resume_pipeline.pkl` is missing.")
    else:
        # Feature Extraction
        char_c, word_c, sent_c, avg_word_len = extract_features(resume_input)
        
        # Text Preprocessing 
        cleaned_text = preprocess_text(resume_input)
        
        # Prediction & Confidence Scoring
        prediction = pipeline.predict([cleaned_text])[0]
        decision_scores = pipeline.decision_function([cleaned_text])[0]
        
        # Softmax calculation to approximate confidence percentage
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / exp_scores.sum()
        confidence_score = np.max(probabilities) * 100
        
        # ATS Pipeline Functions
        found_skills, expected_skills = extract_skills(cleaned_text, prediction)
        ats_score = calculate_ats_score(found_skills, expected_skills)
        missing_skills = get_missing_skills(found_skills, expected_skills)
        suggestions = generate_suggestions(ats_score, missing_skills, word_c, sent_c, avg_word_len)
        
        # ==========================================
        # DASHBOARD RESULTS
        # ==========================================
        st.markdown("---")
        st.markdown("## 📊 Applicant Dashboard")
        
        # Top Row: Prediction & ATS Score
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### 🎯 Predicted Job Category")
            st.markdown(f"<div class='category-highlight'>{prediction.upper()}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='confidence-score'>Model Confidence: {confidence_score:.1f}%</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### 📈 ATS Match Score")
            
            if ats_score >= 75:
                st.markdown(f"<div class='ats-excellent'>{ats_score}%</div>", unsafe_allow_html=True)
            elif ats_score >= 50:
                st.markdown(f"<div class='ats-good'>{ats_score}%</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ats-poor'>{ats_score}%</div>", unsafe_allow_html=True)
            
            st.progress(ats_score / 100.0)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Middle Row: NLP Metrics
        st.markdown("#### 📏 Resume Structural Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Characters", f"{char_c:,}")
        c2.metric("Total Words", f"{word_c:,}")
        c3.metric("Sentence Count", f"{sent_c:,}")
        c4.metric("Avg. Word Length", avg_word_len)

        st.markdown("---")
        
        # Bottom Row: Skill Extraction
        st.markdown("#### 🧩 Skills Extraction & ATS Matching")
        col_found, col_missing = st.columns(2)
        
        with col_found:
            st.markdown("**✅ Detected Core Skills:**")
            if found_skills:
                html_found = "".join([f"<span class='badge-found'>{s.title()}</span>" for s in found_skills])
                st.markdown(html_found, unsafe_allow_html=True)
            else:
                st.info("No primary industry keywords detected.")
                
        with col_missing:
            st.markdown("**❌ Missing Recommended Skills:**")
            if missing_skills:
                html_missing = "".join([f"<span class='badge-missing'>{s.title()}</span>" for s in missing_skills])
                st.markdown(html_missing, unsafe_allow_html=True)
            else:
                st.success("All expected industry keywords are present!")

        # Actionable Suggestions
        st.markdown("---")
        st.markdown("#### 💡 Resume Improvement Suggestions")
        st.markdown("<div class='suggestion-box'>", unsafe_allow_html=True)
        
        for sug in suggestions:
            st.markdown(sug)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data Transparency Expander
        with st.expander("🔍 View Preprocessed NLP Data"):
            st.write(cleaned_text)