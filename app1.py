import pandas as pd
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load and preprocess data
df = pd.read_csv("all-data.csv", encoding='ISO-8859-1')
df.columns = ['sentiment', 'text']

def preprocess(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    return text

df['clean_text'] = df['text'].apply(preprocess)

# 2. Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment'].str.lower()

# 3. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 4. Streamlit UI
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter a sentence to analyze sentiment:")

if st.button("Analyze"):
    cleaned_input = preprocess(user_input)
    vec_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vec_input)[0]
    st.write(f"**Predicted Sentiment:** {prediction.capitalize()}")
