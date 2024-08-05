import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import is_classifier

# Load the model and vectorizer
def load_model_and_vectorizer():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        if not is_classifier(model) or not hasattr(model, 'predict'):
            st.error("Loaded model is not a valid classifier.")
            st.stop()
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        st.stop()

model, vectorizer = load_model_and_vectorizer()

# Function to make predictions
def predict(text):
    try:
        text_transformed = vectorizer.transform([text])
        prediction = model.predict(text_transformed)
        return prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Streamlit app
st.title("EMail/SMS Spam Classifier")

# Input from user
user_input = st.text_area("Enter the text to classify:")

if st.button("Classify"):
    if user_input:
        prediction = predict(user_input)
        if prediction is not None:
            result = "Spam" if prediction == 1 else "Not Spam (Ham)"
            st.write(f"The text is classified as: **{result}**")
    else:
        st.error("Please enter some text to classify.")
