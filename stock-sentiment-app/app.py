import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import numpy as np
import datetime

# Load trained model
model = joblib.load("models/random_forest_sentiment_model.pkl")

# Load sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# App UI
st.set_page_config(page_title="Stock Sentiment Predictor", layout="centered")
st.title("📈 Stock Sentiment Prediction Based on News Headlines")
st.write("Predict whether the stock/index will go **Up or Down** based on today's news sentiment.")

# Text input
headline_input = st.text_area("📰 Enter today's headlines (one per line):", height=200)

if st.button("Predict"):
    if not headline_input.strip():
        st.warning("Please enter at least one headline.")
    else:
        # Split and analyze each headline
        headlines = headline_input.strip().split("\n")
        sentiment_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]

        # Calculate average sentiment
        avg_sentiment = np.mean(sentiment_scores)

        # Dummy inputs for pct_change & rolling average (not used in real-time)
        input_features = [[avg_sentiment, 0.0, avg_sentiment]]

        # Make prediction
        prediction = model.predict(input_features)[0]
        confidence = model.predict_proba(input_features)[0][prediction]

        # Display result
        label = "📈 Stock is likely to go UP" if prediction == 1 else "📉 Stock is likely to go DOWN"
        st.subheader(label)
        st.text(f"Confidence: {confidence * 100:.2f}%")
