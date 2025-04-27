# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

#nltk.download('stopwords')

# Load dataset
data = pd.read_csv('dataset/Reviews.csv')
data = data.sample(n=5000, random_state=42)

# Check basic info
print("First 5 rows:\n", data.head())

# Use only necessary columns
data = data[['Text', 'Score']]

# Optional: Drop NA
data = data.dropna()

# Create Binary Label: Positive (Score >= 4) -> 1, Negative (Score <= 2) -> 0
def label_score(score):
    if score >= 4:
        return 1
    elif score <= 2:
        return 0
    else:
        return None

data['label'] = data['Score'].apply(label_score)
data = data.dropna()  # Drop rows where label is None (Score = 3)

# Data preprocessing
stop_words = set()  # Skip stopword removal for now

def clean_text(text):
    words = text.lower().split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

data['cleaned_text'] = data['Text'].apply(clean_text)

# Feature extraction
# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text']).toarray()  # <--- FITTING done here ✅
y = data['label']

# NOW save this fitted vectorizer correctly

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model and vectorizer BEFORE showing the plot
print("Saving model and vectorizer...")

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# Define file paths
model_path = os.path.join(current_dir, 'saved_model.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

# Save the model
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Save the TF-IDF vectorizer
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"Model saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")
print("Model and Vectorizer saved successfully!")

# Visualization - moved after saving files
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save the plot first
plt.show()  # Then display it (this will pause execution)

print("Script completed successfully!")