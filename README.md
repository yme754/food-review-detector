# 🍔 Food Review Fake Detector (NLP Project)

This project builds a machine learning model to detect fake food reviews using Natural Language Processing (NLP).

## Files
- `main.py`: Train the model and save it.
- `predict.py`: Use the saved model to predict whether a review is genuine or fake.
- `requirements.txt`: List of Python dependencies.
- `dataset/Reviews.csv`: Dataset ([download from Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)).

## How to Run

1. Clone this repository
2. Install dependencies:
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

## License
MIT License
