import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("punkt_tab", quiet=True)
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI-Powered Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# RESOURCE INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource
def download_nltk_dependencies():
    """Download required NLTK resources silently."""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

@st.cache_resource
def load_ml_artifacts():
    """Load the saved TF-IDF vectorizer and Linear SVM model using relative paths."""
    try:
        # Load exactly as they were saved in the notebook
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        model = joblib.load("resume_classifier.pkl")
        return vectorizer, model
    except FileNotFoundError:
        st.error("⚠️ Error: Missing model artifacts. Please ensure 'tfidf_vectorizer.pkl' and 'resume_classifier.pkl' are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred while loading models: {e}")
        st.stop()

# Initialize resources
download_nltk_dependencies()
tfidf_vectorizer, resume_classifier = load_ml_artifacts()

# ==========================================
# EXACT PREPROCESSING LOGIC FROM NOTEBOOK
# ==========================================
def preprocess_text(text):
    """
    Exact text preprocessing function used in the Jupyter Notebook.
    Handles lowercasing, URL removal, punctuation removal, stopword removal, and lemmatization.
    """
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # 3. Remove punctuation and special characters (keep only alphabets)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # 4. Remove extra whitespaces
    text = re.sub(r"\s+", " ", text)
    
    # 5. Remove English stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in text.split() if word not in stop_words]
    
    # 6. Apply WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(words)

# ==========================================
# SIDEBAR SETUP
# ==========================================
with st.sidebar:
    st.title("📄 System Details")
    st.markdown("---")
    
    st.subheader("💡 Project Overview")
    st.write("An automated AI system designed to assist recruiters by evaluating and classifying candidate resumes into suitable job categories based on their skills and textual content.")
    
    st.subheader("📊 Dataset Information")
    st.write("- **Source:** Kaggle Resume Dataset")
    st.write("- **Total Records:** 2,484")
    st.write("- **Target:** 25 Job Categories")
    
    st.subheader("🤖 Model Information")
    st.write("- **Feature Extraction:** TF-IDF")
    st.write("- **Algorithm:** Linear SVM")
    st.write("- **Best Model:** Linear SVM")
    
    st.subheader("🛠️ Technology Stack")
    st.write("Python, Streamlit, Scikit-Learn, NLTK, Pandas")

# ==========================================
# MAIN PAGE INTERFACE
# ==========================================
st.title("AI-Powered Resume Screening System")
st.markdown("Streamline your recruitment process. Paste a candidate's resume below to instantly predict their most suitable job category.")

# Text Input
resume_input = st.text_area(
    "✍️ Paste the resume text here:", 
    height=250, 
    placeholder="e.g. Dedicated Customer Service Manager with 15+ years of experience in Hospitality..."
)

# Predict Button
if st.button("Predict Job Category", type="primary", use_container_width=True):
    if not resume_input.strip():
        st.warning("⚠️ Please enter some resume text to get a prediction.")
    else:
        with st.spinner("Analyzing resume content..."):
            
            # --- Analytics Calculation ---
            char_count = len(resume_input)
            word_count = len(resume_input.split())
            try:
                # Use NLTK for accurate sentence tokenization
                sentence_count = len(nltk.sent_tokenize(resume_input))
            except Exception:
                # Fallback simple sentence split
                sentence_count = resume_input.count('.') + resume_input.count('!') + resume_input.count('?')
            
            # --- ML Pipeline Workflow ---
            # 1. Preprocess the raw text
            cleaned_resume = preprocess_text(resume_input)
            
            # 2. Vectorize using the loaded TF-IDF vectorizer
            vectorized_features = tfidf_vectorizer.transform([cleaned_resume])
            
            # 3. Predict using the loaded Linear SVM model
            predicted_category = resume_classifier.predict(vectorized_features)[0]
            
            # --- Display Results ---
            st.markdown("---")
            st.success(f"### 🎯 Predicted Category: **{predicted_category.upper()}**")
            
            st.markdown("### 📈 Resume Analytics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Character Count:**\n\n {char_count:,}")
            with col2:
                st.info(f"**Word Count:**\n\n {word_count:,}")
            with col3:
                st.info(f"**Sentence Count:**\n\n {sentence_count:,}")
                
            with st.expander("🔍 View Preprocessed (Cleaned) Text"):
                st.write(cleaned_resume)