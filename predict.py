import pickle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'saved_model.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    print("Required files not found.")
    exit(1)

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

review = input("Enter your review: ").strip().lower()
review_clean = ' '.join(review.split())
review_vec = vectorizer.transform([review_clean]).toarray()

prediction = model.predict(review_vec)
prob = model.predict_proba(review_vec)[0]

if prediction[0] == 1:
    print("Prediction: Genuine Review ✅")
else:
    print("Prediction: Fake Review ❌")

print(f"Confidence - Fake: {prob[0]:.4f}, Genuine: {prob[1]:.4f}")
