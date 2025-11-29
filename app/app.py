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
# 2. LOAD MODELS (With Caching for Performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_tools():
    """
    Load the saved model, scaler, and label encoder.
    Using cache to prevent reloading on every user interaction.
    """
    # Get the directory where this current file (app.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths to the 'saved_models' folder relative to this file
    model_path = os.path.join(current_dir, '..', 'saved_models', 'crop_recommendation_model.pkl')
    scaler_path = os.path.join(current_dir, '..', 'saved_models', 'scaler.pkl')
    encoder_path = os.path.join(current_dir, '..', 'saved_models', 'label_encoder.pkl')

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        return model, scaler, encoder
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.error("Please ensure the 'saved_models' folder exists and contains the .pkl files.")
        return None, None, None

model, scaler, le = load_tools()

# -----------------------------------------------------------------------------
# 3. UI LAYOUT
# -----------------------------------------------------------------------------
st.title("🌱 Smart Crop Recommendation System")
st.markdown("""
Welcome! Enter the soil and environmental conditions below to get the 
**most suitable crop** recommendation for your farm.
""")

st.divider() # Adds a visual line separator

if model and scaler and le:
    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50, help="Ratio of Nitrogen content in soil")
        P = st.number_input("Phosphorus (P)", min_value=0, max_value=145, value=50, help="Ratio of Phosphorus content in soil")
        K = st.number_input("Potassium (K)", min_value=0, max_value=205, value=50, help="Ratio of Potassium content in soil")
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    with col2:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)

    # -------------------------------------------------------------------------
    # 4. PREDICTION LOGIC
    # -------------------------------------------------------------------------
    st.markdown("---")
    
    # Center the button
    _, mid_col, _ = st.columns([1, 2, 1])
    
    with mid_col:
        predict_btn = st.button("🔍 Predict Best Crop", type="primary", use_container_width=True)

    if predict_btn:
        # A. Prepare the input array (must match training data order)
        user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # B. Scale the input using the loaded Scaler
        try:
            user_input_scaled = scaler.transform(user_input)
            
            # C. Predict the Class ID
            prediction_id = model.predict(user_input_scaled)
            
            # D. Decode the Class ID to the Crop Name
            predicted_crop_name = le.inverse_transform(prediction_id)[0]
            
            # E. Display Result
            st.success(f"✅ The Recommended Crop is: **{predicted_crop_name.upper()}**")

            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Models could not be loaded. Please check your file structure.")