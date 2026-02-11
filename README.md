# 🧠 Geo-AI News Sentiment Analyzer

A Machine Learning powered web application that analyzes geopolitical news text and predicts sentiment (Positive / Negative) with confidence scoring.

🔗 **Live Demo:** https://geo-ai-news-sentiment-analyzer.streamlit.app

---

## 📌 Project Overview

This project implements an end-to-end NLP pipeline:

- Text preprocessing
- TF-IDF vectorization
- Logistic Regression classification
- Model evaluation (Accuracy, Precision, Recall, F1-score)
- Deployment using Streamlit Cloud

The application allows users to input geopolitical or news-related text and receive:

- Predicted Sentiment
- Model Confidence Score
- Classification behavior based on trained dataset

---

## 🛠️ Tech Stack

- Python 3
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- GitHub
- Streamlit Cloud (Deployment)

---

## 📊 Dataset Details

- Total samples: **5,883**
- Binary classification:
  - Positive
  - Negative
- Combined custom geopolitical data + real labeled dataset

---

## 🤖 Model Details

- Vectorization: **TF-IDF (max_features=5000)**
- Classifier: **Logistic Regression**
- Train-Test Split: 80/20
- Test Accuracy: **75.19%**

### 📈 Classification Report

Accuracy: 75.19%

Precision / Recall / F1-score:
Negative: 0.74 / 0.67 / 0.71
Positive: 0.76 / 0.81 / 0.78


---

## 🧪 Features

- Real-time sentiment prediction
- Confidence score calculation
- Mixed/Uncertain detection for low confidence (<65%)
- Clean UI with sidebar project information
- Fully deployed web application

---

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

To run locally:

```bash
pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

## 👨‍💻 Author

**Prem Nayan**  
B.Tech CSE | AI/ML Enthusiast  
Lovely Professional University  

GitHub: https://github.com/Premnayan14
