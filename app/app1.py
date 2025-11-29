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
# 2. LOAD MODELS (Robust Path Handling & Caching)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_tools():
    """
    Load the saved model, scaler, and label encoder.
    Uses relative paths so it works on any machine (GitHub/Streamlit Cloud).
    """
    # Get the directory where this current file (app.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to project root, then into saved_models
    base_path = os.path.join(current_dir, '..', 'saved_models')
    
    try:
        model = joblib.load(os.path.join(base_path, 'crop_recommendation_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        encoder = joblib.load(os.path.join(base_path, 'label_encoder.pkl'))
        return model, scaler, encoder
    except FileNotFoundError as e:
        st.error(f"❌ Error loading files: {e}")
        st.error(f"Looking in: {base_path}")
        return None, None, None

model, scaler, le = load_tools()

# -----------------------------------------------------------------------------
# 3. UI LAYOUT & INPUTS
# -----------------------------------------------------------------------------
st.title("🌱 Smart Crop Recommendation System")
st.markdown("Enter the soil and environmental conditions below to get the best crop recommendation.")
st.divider()

if model and scaler and le:
    # Use columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧪 Soil Nutrients")
        # Added explicit units in help text
        N = st.number_input("Nitrogen (N)", 0, 140, 50, help="Ratio of Nitrogen content in soil (kg/ha)")
        P = st.number_input("Phosphorus (P)", 0, 145, 50, help="Ratio of Phosphorus content in soil (kg/ha)")
        K = st.number_input("Potassium (K)", 0, 205, 50, help="Ratio of Potassium content in soil (kg/ha)")
        # pH limits strictly enforced 0-14
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, step=0.1, help="0 (Acidic) to 14 (Alkaline)")

    with col2:
        st.subheader("🌤️ Climate Conditions")
        # Added format="%.1f" for cleaner display
        temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, step=0.1, format="%.1f")
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0, step=0.1, format="%.1f")
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0, step=0.1, format="%.1f")

    # -------------------------------------------------------------------------
    # 4. VALIDATION LOGIC (Sanity Checks)
    # -------------------------------------------------------------------------
    warning_msg = ""
    if ph < 4.0 or ph > 9.0:
        warning_msg += "⚠️ **Warning:** Your Soil pH is extremely acidic/alkaline. Most crops prefer pH 5.5-7.5.\n\n"
    if temperature > 45:
        warning_msg += "⚠️ **Warning:** Temperature is extremely high (>45°C). Ensure crops are heat-tolerant.\n\n"
    if humidity < 10:
        warning_msg += "⚠️ **Warning:** Humidity is extremely low (<10%). Intensive irrigation required.\n\n"

    if warning_msg:
        st.warning(warning_msg)

    # -------------------------------------------------------------------------
    # 5. PREDICTION LOGIC
    # -------------------------------------------------------------------------
    st.markdown("---")
    _, mid_col, _ = st.columns([1, 2, 1])
    
    with mid_col:
        predict_btn = st.button("🔍 Predict Best Crop", type="primary", use_container_width=True)

    if predict_btn:
        try:
            # A. Prepare Input
            user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            
            # B. Scale Input
            user_input_scaled = scaler.transform(user_input)
            
            # C. Predict Class ID
            prediction_id = model.predict(user_input_scaled)
            
            # D. Decode Label to Name
            predicted_crop_name = le.inverse_transform(prediction_id)[0]
            
            # E. Success Output
            st.success(f"✅ The Recommended Crop is: **{predicted_crop_name.upper()}**")
            
            # F. Contextual Advice (Adds value for the user)
            crop_lower = predicted_crop_name.lower()
            if crop_lower in ['rice', 'jute', 'papaya', 'coconut', 'coffee']:
                st.info("ℹ️ Note: This crop generally requires high water availability/rainfall.")
            elif crop_lower in ['chickpea', 'mothbeans', 'kidneybeans', 'blackgram', 'lentil']:
                st.info("ℹ️ Note: This crop is relatively drought-resistant.")
            elif crop_lower in ['cotton']:
                st.info("ℹ️ Note: Cotton requires long frost-free periods and plenty of sunshine.")
            elif crop_lower in ['banana']:
                st.info("ℹ️ Note: Banana requires high humidity and moisture.")
            elif crop_lower in ['maize']:
                st.info("ℹ️ Note: Maize requires well-drained fertile soil.")
                
            st.balloons()
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

else:
    st.warning("⚠️ Models could not be loaded. Please check your 'saved_models' folder.")