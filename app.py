import streamlit as st
import pickle
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI & ML News Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

# ================= LOAD MODEL =================
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ================= SIDEBAR =================
st.sidebar.title("📌 Project Information")

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Prem Nayan")

st.sidebar.markdown("### 🤖 Model Used")
st.sidebar.write("TF-IDF + Logistic Regression")

st.sidebar.markdown("### 📊 Model Accuracy")
st.sidebar.write("75.19%")

st.sidebar.markdown("### 📁 Dataset Size")
st.sidebar.write("5,883 samples")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Python & Streamlit")

# ================= HEADER =================
st.markdown(
    "<h1 style='text-align: center;'>🧠 AI & ML News Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Analyze geopolitical or news text using a trained Machine Learning model</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ================= INPUT =================
st.markdown("## 📰 Enter News Text")

user_text = st.text_area(
    "Paste a news sentence or short paragraph below:",
    height=150,
    placeholder="Example: Peace talks between the two countries were successful..."
)

analyze = st.button("🔍 Analyze Sentiment")

# ================= RESULT =================
if analyze:
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        text_vector = vectorizer.transform([user_text])
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        confidence = np.max(probabilities) * 100

        st.markdown("## 📊 Analysis Result")

        if confidence < 65:
            st.markdown(
                "<h3 style='color:orange;'>⚠️ Sentiment: MIXED / UNCERTAIN</h3>",
                unsafe_allow_html=True
            )
        elif prediction.lower() == "positive":
            st.markdown(
                "<h3 style='color:green;'>✅ Sentiment: POSITIVE</h3>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h3 style='color:red;'>❌ Sentiment: NEGATIVE</h3>",
                unsafe_allow_html=True
            )

        st.write(f"🔍 Model Confidence: {confidence:.2f}%")

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: gray;'>Deployed using Streamlit Cloud</p>",
    unsafe_allow_html=True
)
