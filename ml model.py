import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load the preprocessed sentiment dataset
df = pd.read_csv("data/final_sentiment_data.csv")

# ─────────────────────────────────────────────
# ✅ ENHANCEMENT 1: Add new features
# ─────────────────────────────────────────────

# 1. Percentage change in closing price
df['pct_change'] = df['CP'].pct_change()

# 2. Rolling 3-day average of sentiment score
df['sentiment_3d'] = df['sentiment'].rolling(window=3).mean()

# Drop rows with NaNs caused by pct_change & rolling
df.dropna(inplace=True)

# ─────────────────────────────────────────────
# 🎯 Features & Labels
# ─────────────────────────────────────────────
X = df[['sentiment', 'pct_change', 'sentiment_3d']]
y = df['target']

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─────────────────────────────────────────────
# ✅ ENHANCEMENT 2: Random Forest with class balancing
# ─────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 📊 Evaluation
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%\n")

print("📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 💾 Save Model (optional)
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest_sentiment_model.pkl")
print("✅ Model saved to 'models/random_forest_sentiment_model.pkl'")
