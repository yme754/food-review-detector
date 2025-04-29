import streamlit as st
import pickle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'saved_model.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

st.title("🍔 Genuine vs Fake Food Review Detector")
st.write("Check if a review sounds genuine or fake")

review = st.text_area("Type your food review here:")

if st.button("Check Review"):
    if not review.strip():
        st.warning("Please write a review first.")
    else:
        processed_review = ' '.join(review.lower().split())
        vector = vectorizer.transform([processed_review]).toarray()
        prediction = model.predict(vector)
        prob = model.predict_proba(vector)[0]

        if prediction[0] == 1:
            st.success("✅ This seems like a genuine review!")
        else:
            st.error("❌ This review might be fake!")

        st.subheader("Confidence Levels")
        st.write(f"Fake: {prob[0]:.4f}")
        st.write(f"Genuine: {prob[1]:.4f}")
