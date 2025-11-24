import streamlit as st
import joblib
import numpy as np

# Load model, scaler, and label encoder
model = joblib.load("../saved_models/crop_recommendation_model.pkl")
scaler = joblib.load("../saved_models/scaler.pkl")
le = joblib.load("../saved_models/label_encoder.pkl")

st.set_page_config(page_title="Smart Crop Recommendation", page_icon="🌱")
st.title("🌱 Smart Crop Recommendation System")
st.write("Enter soil and environmental conditions to get the recommended crop.")

# User inputs
N = st.number_input("Nitrogen (N)", 0, 140, 50)
P = st.number_input("Phosphorus (P)", 5, 145, 50)
K = st.number_input("Potassium (K)", 5, 205, 50)
temperature = st.number_input("Temperature (°C)", 0, 50, 25)
humidity = st.number_input("Humidity (%)", 0, 100, 50)
ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

if st.button("Predict Crop"):
    user_input = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)
    predicted_crop = le.inverse_transform(prediction)[0]
    st.success(f"✅ Recommended Crop: **{predicted_crop.title()}**")
