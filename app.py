import streamlit as st
import pickle
import os

# Load model and vectorizer
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'saved_model.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit App
st.title("🍔 Food Review Fake Detector")
st.write("Detect if a review is genuine or fake using NLP and Machine Learning.")

review = st.text_area("Enter your food review:")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Preprocess
        review_clean = review.lower().split()
        review_clean = ' '.join(review_clean)

        # Transform
        review_vec = vectorizer.transform([review_clean]).toarray()

        # Predict
        prediction = model.predict(review_vec)
        prob = model.predict_proba(review_vec)[0]

        if prediction[0] == 1:
            st.success("✅ Prediction: Positive / Genuine Review")
        else:
            st.error("❌ Prediction: Negative / Fake Review")

        st.subheader("Confidence Scores:")
        st.write(f"- Negative: {prob[0]:.4f}")
        st.write(f"- Positive: {prob[1]:.4f}")
