import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Load dataset
df = pd.read_csv("data/sp500_headlines_2008_2024.csv")

# Convert date to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Group all headlines by date
grouped = df.groupby('Date')['Title'].apply(lambda x: ' '.join(x)).reset_index()

# Apply VADER sentiment analysis
analyzer = SentimentIntensityAnalyzer()
grouped['sentiment'] = grouped['Title'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Merge with closing prices
grouped = grouped.merge(df[['Date', 'CP']].drop_duplicates(), on='Date')

# Sort by date
grouped = grouped.sort_values('Date').reset_index(drop=True)

# Create target column: 1 if next day's CP is higher, else 0
grouped['Next_CP'] = grouped['CP'].shift(-1)
grouped['target'] = (grouped['Next_CP'] > grouped['CP']).astype(int)

# Drop the last row with NaN
grouped.dropna(inplace=True)

# Save final file
os.makedirs("data", exist_ok=True)
grouped.to_csv("data/final_sentiment_data.csv", index=False)

print("✅ Preprocessed data saved to 'data/final_sentiment_data.csv'")
