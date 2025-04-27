import pickle
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# Define file paths
model_path = os.path.join(current_dir, 'saved_model.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

# Check if the files exist
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    print(f"Files found at:")
    print(f"- Model: {model_path}")
    print(f"- Vectorizer: {vectorizer_path}")
else:
    print("Files not found at:")
    print(f"- Model: {model_path}")
    print(f"- Vectorizer: {vectorizer_path}")
    exit(1)  # Exit if files aren't found

# Load model and vectorizer
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Take custom review
review = input("Enter your review: ")
# Preprocess
review_clean = review.lower().split()
review_clean = ' '.join(review_clean)  # No stopwords removal since set() was empty

# Transform
review_vec = vectorizer.transform([review_clean]).toarray()

# Predict
prediction = model.predict(review_vec)

if prediction[0] == 1:
    print("Prediction: Positive/Genuine Review ✅")
else:
    print("Prediction: Negative/Fake Review ❌")

# Show prediction probability
prob = model.predict_proba(review_vec)[0]
print(f"Confidence scores: Negative: {prob[0]:.4f}, Positive: {prob[1]:.4f}")