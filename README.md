# 📄 AI-Powered Resume Screening System

An NLP-based Machine Learning application that automatically classifies resumes into predefined job categories using TF-IDF and Linear SVM.

---

## 🚀 Project Overview

The AI-Powered Resume Screening System is designed to automate the resume categorization process. It uses Natural Language Processing (NLP) techniques to preprocess resume text and predict the most suitable job category.

This helps reduce manual screening effort and enables faster candidate classification.

---

## 🎯 Objectives

- Automate resume categorization
- Apply NLP techniques on unstructured text
- Compare multiple Machine Learning models
- Select the best-performing classifier
- Deploy the solution using Streamlit

---

## 📊 Dataset Information

- **Source:** Kaggle Resume Dataset
- **Total Records:** 2,484 Resumes
- **Feature:** Resume Text
- **Target:** Job Category
- **Problem Type:** Multi-Class Text Classification

### Categories Include

- Information Technology
- Engineering
- Finance
- HR
- Sales
- Business Development
- Healthcare
- Teacher
- Advocate
- Chef
- Designer
- Banking
- Aviation
- Consultant
- And more...

---

## 🛠️ Technology Stack

### Programming Language
- Python

### Data Analysis
- Pandas
- NumPy

### NLP
- NLTK

### Machine Learning
- Scikit-Learn

### Deployment
- Streamlit

---

## 🔄 Project Workflow

```text
Resume Text
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Model Training
      │
      ▼
Linear SVM Classifier
      │
      ▼
Category Prediction
```

---

## 🧹 NLP Preprocessing

The resume text undergoes:

- Lowercasing
- URL Removal
- Special Character Removal
- Stopword Removal
- Lemmatization
- Whitespace Cleaning

---

## 📈 Feature Engineering

Additional text features were created:

- Character Count
- Word Count
- Sentence Count
- Average Word Length
- Stopword Count
- Digit Count
- Capital Letter Count
- Punctuation Count

---

## 🤖 Models Evaluated

| Model | Evaluated |
|---------|---------|
| Multinomial Naive Bayes | ✅ |
| Logistic Regression | ✅ |
| Linear SVM | ✅ |

### Best Model

**Linear Support Vector Machine (SVM)**

Reason:
- Better generalization
- Strong performance on high-dimensional text data
- Best balance between Precision and Recall

---

## 📊 Model Evaluation

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The Linear SVM model achieved the best overall performance among all evaluated models.

---

## 🌐 Streamlit Application

Features:

✅ Resume Category Prediction

✅ Resume Analytics

- Character Count
- Word Count
- Sentence Count

✅ Clean and Interactive User Interface

---

## 📁 Project Structure

```text
Resume_ATS_Project/
│
├── dataset/
├── models/
├── plots/
├── Resume_ATS_System.ipynb
├── streamlit_app.py
├── resume_classifier.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone Repository

```bash
git clone https://github.com/SaiSourav2004/Resume-Classification-Using-NLP.git
```

Move to Project Directory

```bash
cd Resume-Classification-Using-NLP
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run Application

```bash
streamlit run streamlit_app.py
```

---

## 🔮 Future Enhancements

- Resume PDF Upload
- DOCX Resume Parsing
- Skill Extraction
- Job Description Matching
- ATS Compatibility Score
- Resume Improvement Suggestions
- BERT-Based Classification
- AI-Powered Recruitment Assistant

---

## 👨‍💻 Author

### Sai Sourav Panigrahi

- Data Science & Machine Learning Enthusiast
- Python Developer
- NLP Projects

GitHub:
https://github.com/SaiSourav2004

---

⭐ If you found this project useful, consider giving it a Star.
