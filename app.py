import streamlit as st
import pickle

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI & ML News Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

# ================= LOAD MODEL =================
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

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

with st.container():
    st.markdown(
        "<div style='padding:15px; border-radius:10px; border:1px solid #ddd;'>",
        unsafe_allow_html=True
    )

    user_text = st.text_area(
        "Paste a news sentence or short paragraph below:",
        height=150,
        placeholder="Example: Peace talks between the two countries were successful..."
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ================= BUTTON =================
analyze = st.button("🔍 Analyze Sentiment")

# ================= RESULT =================
if analyze:
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        # Vectorize input
        text_vector = vectorizer.transform([user_text])

        # Prediction + confidence
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities) * 100

        st.markdown("## 📊 Analysis Result")

        with st.container():
            st.markdown(
                "<div style='padding:20px; border-radius:10px; background-color:#f9f9f9; border:1px solid #ddd;'>",
                unsafe_allow_html=True
            )

            # ===== SENTIMENT LOGIC =====
            if confidence < 65:
                st.markdown(
                    "<h3 style='color:orange; text-align:center;'>⚠️ Sentiment: MIXED / UNCERTAIN</h3>",
                    unsafe_allow_html=True
                )
            elif prediction.lower() == "positive":
                st.markdown(
                    "<h3 style='color:green; text-align:center;'>✅ Sentiment: POSITIVE</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<h3 style='color:red; text-align:center;'>❌ Sentiment: NEGATIVE</h3>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"<p style='text-align:center; font-size:16px;'>🔍 Model Confidence: <b>{confidence:.2f}%</b></p>",
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: gray;'>AI & ML Project | Built with Python & Streamlit</p>",
    unsafe_allow_html=True
)
