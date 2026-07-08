# Smart Resume Analyzer and ATS System using NLP and Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end NLP and Machine Learning application that automates resume screening by classifying resumes into job categories, extracting skills, calculating ATS scores, identifying skill gaps, and providing actionable resume improvement suggestions through an interactive Streamlit dashboard.

---

# 🚀 Project Overview

Recruiters often receive hundreds of resumes for a single job opening, making manual screening time-consuming and inefficient. This project streamlines the initial screening process by leveraging Natural Language Processing (NLP) and Machine Learning techniques.

The system analyzes uploaded resumes, predicts the most relevant job category, evaluates resume quality using ATS-inspired metrics, identifies missing skills, and generates personalized suggestions to improve resume effectiveness.

---

# ✨ Key Features

### 📄 Resume Classification

* Multi-class resume classification across 24 job categories
* NLP-based text preprocessing
* TF-IDF feature extraction
* Linear SVM classification
* Confidence score estimation

### 📂 Resume Upload Support

* PDF Resume Upload
* DOCX Resume Upload
* TXT Resume Upload
* Direct Text Input

### 🎯 ATS Enhancement Module

* Skills Extraction
* ATS Score Calculation
* Missing Skills Detection
* Resume Improvement Suggestions

### 🖥️ Interactive Dashboard

* Streamlit-based Web Application
* Dark Theme User Interface
* Real-Time Resume Analysis
* User-Friendly Layout

---

# 📊 Dataset Overview

| Attribute      | Value                           |
| -------------- | ------------------------------- |
| Dataset Source | Kaggle Resume Dataset           |
| Total Records  | 2,484 Resumes                   |
| Job Categories | 24                              |
| Target Column  | Category                        |
| Problem Type   | Multi-Class Text Classification |

### Features Used

```text
Resume_str
Resume_html
Clean_Resume
```

### Target Variable

```text
Category
```

---

# 🛠️ Technology Stack

| Category                    | Technology             |
| --------------------------- | ---------------------- |
| Programming Language        | Python                 |
| Machine Learning            | Scikit-Learn           |
| Natural Language Processing | NLTK                   |
| Feature Engineering         | TF-IDF Vectorization   |
| Classification Algorithm    | Linear SVM (LinearSVC) |
| Web Framework               | Streamlit              |
| File Processing             | PyPDF2, python-docx    |
| Model Serialization         | Joblib                 |

---

# 🔄 Project Workflow

```text
Resume Upload
      ↓
Text Extraction
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Linear SVM Classification
      ↓
Resume Category Prediction
      ↓
Skills Extraction
      ↓
ATS Score Calculation
      ↓
Missing Skills Detection
      ↓
Resume Suggestions
      ↓
Streamlit Dashboard
```

---

# 🧠 Machine Learning Pipeline

The final model is implemented using a Scikit-Learn Pipeline that combines TF-IDF feature extraction and Linear SVM classification.

```python
resume_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        sublinear_tf=True
    )),
    ('classifier', LinearSVC())
])
```

### Saved Model

```text
resume_pipeline.pkl
```

### Why TF-IDF?

TF-IDF transforms resume text into numerical vectors by assigning greater importance to meaningful terms while reducing the impact of frequently occurring words.

### Why Linear SVM?

Linear SVM performs exceptionally well on high-dimensional sparse text data, making it highly effective for text classification problems such as resume categorization.

### Why Pipeline?

Using a machine learning pipeline ensures that preprocessing and classification steps remain consistent during training, testing, and deployment.

---

# 🎯 ATS Enhancement Module

To improve the practical usefulness of the system, an ATS Enhancement Module was integrated after the resume classification stage.

## ATS Features

### Skills Extraction

Extracts relevant technical and professional skills from resume content.

```python
extract_skills()
```

### ATS Score Calculation

Generates an ATS-inspired score based on resume quality indicators.

```python
calculate_ats_score()
```

### Missing Skills Detection

Identifies important skills not present in the resume.

```python
recommended_skills
```

### Resume Suggestions

Provides personalized recommendations for resume improvement.

```python
generate_suggestions()
```

---

## ATS Score Logic

> **Important Note:** The current ATS score is heuristic-based and is not compared against a specific Job Description (JD).

The ATS score is calculated using:

* Skills Found
* Resume Length
* Education Keywords
* Experience Keywords
* Project Keywords

This approach evaluates overall resume quality and completeness.

### Future Enhancement

Job Description (JD)-based ATS scoring is planned as future scope.

---

# 🖥️ Streamlit Application

The project includes a fully interactive Streamlit application that allows users to upload resumes and receive instant analysis.

### Available Features

✅ Resume Classification

✅ PDF Upload

✅ DOCX Upload

✅ TXT Upload

✅ Skills Extraction

✅ ATS Score

✅ Missing Skills Detection

✅ Resume Suggestions

✅ Confidence Score

✅ Dark Theme UI

✅ Resume Logo Sidebar

---

# 🏗️ Project Architecture

```text
+---------------------------------------------------+
|               Resume Upload Module                |
|             (PDF / DOCX / TXT Files)              |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|              Text Extraction Module               |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|           NLP Text Preprocessing Layer            |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|         TF-IDF Feature Vectorization              |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|        Linear SVM Resume Classification           |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|           Resume Category Prediction              |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|              ATS Enhancement Module               |
| Skills | ATS Score | Missing Skills | Suggestions |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|             Streamlit Dashboard UI                |
+---------------------------------------------------+
```

---

# ⚙️ Installation Guide

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Smart-Resume-Analyzer-and-ATS-System.git
cd Smart-Resume-Analyzer-and-ATS-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run streamlit_app.py
```

---

# 🚀 Usage

1. Launch the Streamlit application.
2. Upload a resume in PDF, DOCX, or TXT format.
3. The system extracts and preprocesses the resume text.
4. The trained machine learning model predicts the resume category.
5. Skills are extracted from the resume.
6. ATS score is calculated.
7. Missing skills are identified.
8. Improvement suggestions are displayed.
9. Results are presented through the Streamlit dashboard.

---

# 📁 Project Structure

```text
Smart-Resume-Analyzer-and-ATS-System/
│
├── Resume_ATS_System.ipynb
├── streamlit_app.py
├── resume_pipeline.pkl
├── requirements.txt
├── resume_logo.jpeg
├── README.md
│
├── dataset/
│   └── UpdatedResumeDataSet.csv
│
└── screenshots/
    ├── home.png
    ├── prediction.png
    └── ats_score.png
```

---

# 📸 Screenshots

## 🏠 Home Page

*Add screenshot here*

---

## 📊 Resume Prediction

*Add screenshot here*

---

## 🎯 ATS Analysis Dashboard

*Add screenshot here*

---

# 📈 Results

The developed system successfully demonstrates:

* Automated Resume Classification
* NLP-Based Resume Processing
* ATS-Oriented Resume Evaluation
* Skill Extraction and Gap Analysis
* Real-Time Resume Assessment
* Interactive Dashboard-Based Deployment

The project combines machine learning and practical resume analysis capabilities within a single application.

---

# 🔮 Future Scope

* Job Description (JD)-Based ATS Scoring
* Resume Ranking System
* Resume-to-Job Matching Engine
* Transformer Models (BERT, RoBERTa)
* Semantic Skill Matching
* Cloud Deployment
* Multi-Language Resume Analysis
* AI-Based Resume Rewriting Suggestions

---

# 🎓 Key Learning Outcomes

* Natural Language Processing (NLP)
* Text Cleaning and Preprocessing
* TF-IDF Feature Engineering
* Machine Learning Pipelines
* Linear Support Vector Machines
* Multi-Class Text Classification
* Resume Analysis Systems
* ATS-Based Evaluation
* Streamlit Application Development
* Model Deployment and Serialization

---

# 👨‍💻 Author

**Sai Sourav Panigrahi**

B.Tech – Computer Science & Engineering

Machine Learning | Data Science | NLP Enthusiast

---

# 📜 License

This project is licensed under the MIT License.

Feel free to use, modify, and enhance this project for learning and research purposes.
