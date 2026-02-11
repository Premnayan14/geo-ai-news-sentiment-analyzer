import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# ================= LOAD DATA =================
# df = pd.read_csv("data.csv")
df = pd.read_csv("final_data.csv")


# Ensure correct columns
df.columns = ["text", "label"]

# Clean text
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip().str.len() > 5]

# Split features and labels
X = df["text"].values
y = df["label"].values

# ================= VECTORIZE =================
# vectorizer = TfidfVectorizer(
#     max_features=10000,
#     lowercase=True
# )
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    lowercase=True
)


X_vectorized = vectorizer.fit_transform(X)

# ================= TRAIN MODEL =================
model = LogisticRegression(max_iter=3000)
model.fit(X_vectorized, y)

# ================= SAVE MODEL =================
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# ================= CONFIRMATION =================
print(f"✅ ML model trained successfully on {len(X)} samples")
