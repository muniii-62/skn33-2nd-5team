import joblib
import streamlit as st
from config import MODEL_PATH, PREPROCESSOR_PATH

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor