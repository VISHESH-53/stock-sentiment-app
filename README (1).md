# 🧠 Stock Sentiment Analysis App

A Streamlit-based web application that performs sentiment analysis on stock-related tweets using a trained machine learning model and VADER sentiment analysis.

## 🚀 Features

- 📊 Predicts sentiment of tweets (Positive, Negative, Neutral)
- 🤖 Uses Random Forest model trained on tweet data
- 💬 Integrates VADER sentiment analyzer
- 🎯 Clean and interactive Streamlit UI

## 🧰 Tech Stack

- Python
- Streamlit
- Scikit-learn
- VADER Sentiment
- Joblib
- Pandas / Numpy

## 📁 Project Structure

```
stock-sentiment-app/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── random_forest_sentiment_model.pkl
└── assets/
    └── screenshot.png (optional)
```

## ▶️ Run Locally

1. Clone the repo:
   ```
   git clone https://github.com/VISHESH-53/stock-sentiment-app.git
   cd stock-sentiment-app
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

## 📸 Preview

![Screenshot](assets/screenshot.png)

## 📄 License

This project is licensed under the MIT License.
