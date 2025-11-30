import streamlit as st
import joblib
import numpy as np
import os
from PIL import Image

# ---------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌱",
    layout="centered"
)

# ---------------------------------------------------------------------
# 2. LOAD MODELS WITH CACHING
# ---------------------------------------------------------------------
@st.cache_resource
def load_tools():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, '..', 'saved_models')

    try:
        model = joblib.load(os.path.join(base_path, 'crop_recommendation_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        encoder = joblib.load(os.path.join(base_path, 'label_encoder.pkl'))
        return model, scaler, encoder
    except FileNotFoundError as e:
        st.error(f"❌ Error loading model files: {e}")
        st.error(f"Checked directory: {base_path}")
        return None, None, None

model, scaler, le = load_tools()

# ---------------------------------------------------------------------
# 3. APP UI
# ---------------------------------------------------------------------
st.title("🌱 Smart Crop Recommendation System")
st.markdown("Provide your soil and climate information to receive the best crop recommendation.")
st.divider()

# Absolute path to images folder
images_path = r"C:\Users\Admin\OneDrive - United States International University (USIU)\Documents\USIU_A\FS2025\DSA3020VA\Crop\images"

if model and scaler and le:

    col1, col2 = st.columns(2)

    # ---------------- INPUTS ----------------
    with col1:
        st.subheader("🧪 Soil Nutrients (kg/ha)")
        N = st.number_input("Nitrogen (N)", 0, 140, 50)
        P = st.number_input("Phosphorus (P)", 0, 145, 50)
        K = st.number_input("Potassium (K)", 0, 205, 50)

    with col2:
        st.subheader("🌤️ Climate Conditions")
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0, step=0.1)

    st.markdown("---")

    # Predict Button
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        predict_btn = st.button("🔍 Predict Best Crop", use_container_width=True)

    if predict_btn:
        try:
            # ---------------- PREDICTION ----------------
            user_input = np.array([[K, humidity, rainfall, N, P]])
            user_input_scaled = scaler.transform(user_input)
            prediction_id = model.predict(user_input_scaled)
            crop_name = le.inverse_transform(prediction_id)[0]

            # Show crop name
            st.success(f"✅ Recommended Crop: **{crop_name.upper()}**")
            st.balloons()

            # ---------------- SHOW CROP IMAGE ----------------
            # Image file assumed to be named exactly as the crop in lowercase
            img_file = os.path.join(images_path, f"{crop_name.lower()}.jpg")
            if os.path.exists(img_file):
                img = Image.open(img_file)
                st.image(img, caption=f"{crop_name.upper()}", use_column_width=True)
            else:
                st.warning(f"⚠️ Image for {crop_name} not found at {img_file}")

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

else:
    st.warning("⚠️ Model files could not be loaded. Check the 'saved_models' folder.")
