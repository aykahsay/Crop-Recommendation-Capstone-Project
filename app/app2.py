import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌱",
    layout="centered"
)

# -----------------------------------------------------------------------------
# 2. LOAD MODELS WITH CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_tools():
    """
    Load the saved model, scaler, and label encoder.
    Uses relative paths for Streamlit Cloud & local deployment.
    """
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

# -----------------------------------------------------------------------------
# 3. APP UI
# -----------------------------------------------------------------------------
st.title("🌱 Smart Crop Recommendation System")
st.markdown("Provide your soil and climate information to receive the best crop recommendation.")
st.divider()

if model and scaler and le:

    col1, col2 = st.columns(2)

    # ------------------------ INPUTS ------------------------
    with col1:
        st.subheader("🧪 Soil Nutrients (kg/ha)")
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
        P = st.number_input("Phosphorus (P)", min_value=0, max_value=145, value=50)
        K = st.number_input("Potassium (K)", min_value=0, max_value=205, value=50)

    with col2:
        st.subheader("🌤️ Climate Conditions")
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)

    st.markdown("---")

    # Predict Button
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        predict_btn = st.button("🔍 Predict Best Crop", use_container_width=True)

    # -----------------------------------------------------------------------------
    # 4. PREDICTION LOGIC (Correct Feature Order)
    # -----------------------------------------------------------------------------
    if predict_btn:
        try:
            # -------------------------------------------------------------
            # IMPORTANT: Your model was trained on EXACTLY this order:
            # ['Potassium', 'humidity', 'rainfall', 'Nitrogen', 'Phosphorous']
            # -------------------------------------------------------------
            user_input = np.array([[
                K,          # Potassium
                humidity,   # humidity
                rainfall,   # rainfall
                N,          # Nitrogen
                P           # Phosphorous
            ]])

            # Scale user input
            user_input_scaled = scaler.transform(user_input)

            # Predict encoded label
            prediction_id = model.predict(user_input_scaled)

            # Decode encoded label → crop name
            crop_name = le.inverse_transform(prediction_id)[0]

            # Output final result
            st.success(f"✅ Recommended Crop: **{crop_name.upper()}**")
            st.balloons()

            # Optional advice
            crop_lower = crop_name.lower()
            if crop_lower in ['rice', 'jute', 'papaya', 'coconut', 'coffee']:
                st.info("💧 This crop requires high water availability.")
            elif crop_lower in ['chickpea', 'mothbeans', 'kidneybeans', 'blackgram', 'lentil']:
                st.info("🔥 This crop is drought-resistant.")
            elif crop_lower == 'cotton':
                st.info("🌞 Cotton requires long frost-free periods and sunshine.")
            elif crop_lower == 'banana':
                st.info("🌫 Banana needs high humidity and moisture.")
            elif crop_lower == 'maize':
                st.info("🌽 Maize grows best in well-drained fertile soil.")

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

else:
    st.warning("⚠️ Model files could not be loaded. Check the 'saved_models' folder.")
