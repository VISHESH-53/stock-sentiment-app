import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load the preprocessed CSV
df = pd.read_csv("data/final_sentiment_data.csv")

# Add features
df['pct_change'] = df['CP'].pct_change()
df['sentiment_3d'] = df['sentiment'].rolling(3).mean()
df.dropna(inplace=True)

# Features and target
X = df[['sentiment', 'pct_change', 'sentiment_3d']]
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train Random Forest with balanced class weights
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Create 'models' folder if not exists
os.makedirs("models", exist_ok=True)

# Save the model
joblib.dump(model, "models/random_forest_sentiment_model.pkl")
print("✅ Model saved to models/random_forest_sentiment_model.pkl")
