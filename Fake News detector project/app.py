import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords if not already
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load model and vectorizer
with open("news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Check whether a news article is *Fake* or *Real* using AI.")

user_input = st.text_area("🧾 Enter the news article here:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vectorized)[0]
    if prediction == 1:
        st.success("✅ This news article is *Real*.")
    else:
        st.error("⚠️ This news article is *Fake*.")