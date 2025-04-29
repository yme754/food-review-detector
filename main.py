import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

data = pd.read_csv('dataset/Reviews.csv').sample(n=5000, random_state=42)
data = data[['Text', 'Score']].dropna()

def label_review(score):
    if score >= 4:
        return 1
    elif score <= 2:
        return 0
    return None

data['label'] = data['Score'].apply(label_review)
data = data.dropna()

def preprocess(text):
    return ' '.join(text.lower().split())

data['cleaned_text'] = data['Text'].apply(preprocess)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text']).toarray()
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

base_path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_path, 'saved_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(base_path, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
