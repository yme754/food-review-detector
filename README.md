# 🍔 Genuine vs Fake Food Review Detector

This project helps detect whether a food review is genuine or possibly fake using Natural Language Processing (NLP) and Machine Learning (ML).

## Project Structure
- `main.py`: Trains a logistic regression model and saves it.
- `predict.py`: A simple CLI tool to check if a review is fake or genuine.
- `app.py`: Streamlit-based web app to interact with the model.
- `requirements.txt`: List of Python dependencies.
- `dataset/Reviews.csv`: Dataset ([download from Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)).

## Run Locally

1. Clone this repository
   ```bash
   https://github.com/yme754/food-review-detector.git
   ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Train the model:
    ```bash
    python main.py
    ```
4. Make predictions:
    ```bash
    python predict.py
    ```
   
## Notes
- The model uses TF-IDF for feature extraction and Logistic Regression for classification.
- Only reviews with ratings 4–5 (positive) and 1–2 (negative) are used. Neutral reviews are ignored.
