import streamlit as st
import pandas as pd
from pathlib import Path

# We import the prediction function from the file we already created
# This is a very clean way to separate the UI from the model logic
try:
    from model_pipeline import predict_pipeline
except ImportError:
    st.error("Error: 'model_pipeline.py' not found. Please make sure it's in the same folder as app.py.")
    st.stop()


# --- 1. SET UP FILE PATH (RELATIVE PATH) ---
# This path is relative to the app.py file
# This will work locally AND in Docker
logo_path = "az-logo.png"

# --- 2. App Configuration ---
st.set_page_config(
    page_title="Al-Zahra Attendance Predictor",
    page_icon=logo_path,  # Use the relative path
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 3. App Header (THE FIX) ---
# Use columns to put the logo and title side-by-side
col1, col2 = st.columns([1, 4]) # 1 part logo, 4 parts title

with col1:
    try:
        # Use st.image to display the logo
        st.image(logo_path, width=100) # Adjust width as needed
    except st.errors.StreamlitAPIException:
        st.error("Logo file not found. Make sure 'az-logo.png' is in the same folder as app.py.")

with col2:
    # st.title() only displays text, not images
    st.title("Al-Zahra Event Predictor")

st.markdown("Use the form in the sidebar to enter event details and predict the attendance.")


# --- Sidebar Inputs ---
st.sidebar.header("Event Features")

# Get the list of program types
# IMPORTANT: Update this list to match all the unique types in your data
program_type_options = [
    "small",
    "medium",
    "large",
    "extra large"
]

program_type = st.sidebar.selectbox(
    "Program Type:",
    options=program_type_options,
    help="Select the type of program."
)

is_weekend = st.sidebar.toggle(
    "Is it a weekend?",
    value=True,
    help="Check this if the event is on a Saturday or Sunday."
)

is_summer = st.sidebar.toggle(
    "Is it in summer?",
    value=False,
    help="Check this if the event is in June, July, or August."
)

is_food = st.sidebar.toggle(
    "Is food being served?",
    value=True,
    help="Check this if food (e.g., dinner, Iftar) will be provided."
)

special_speaker_flag = st.sidebar.toggle(
    "Is there a special speaker?",
    value=False,
    help="Check this if a well-known or guest speaker is featured."
)

is_special_month = st.sidebar.toggle(
    "Is it a special Islamic month?",
    value=True,
    help="Check this for months like Muharram, Ramadan, etc."
)

# --- Prediction Logic ---
if st.sidebar.button("Predict Attendance", type="primary", use_container_width=True):
    
    # 1. Collect all inputs into the dictionary that predict_pipeline expects
    input_data = {
        "is_summer": 1 if is_summer else 0,
        "is_weekend": 1 if is_weekend else 0,
        "program_type": program_type,
        "special_speaker_flag": 1 if special_speaker_flag else 0,
        "is_special_month": 1 if is_special_month else 0,
        "is_food": 1 if is_food else 0
    }

    # 2. Call the prediction pipeline
    try:
        prediction_output = predict_pipeline(input_data)
        predicted_attendance = prediction_output.get("predicted_attendance")

        # 3. Display the result
        st.subheader("📈 Predicted Attendance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Predicted Guests",
                value=f"{predicted_attendance:.0f}"
            )
        
        st.info(
            f"**Confidence Range:** Based on the model's R² score of 0.68 and an average error (MAE) of ~6, you can "
            f"confidently plan for an attendance between **{predicted_attendance - 6:.0f} and {predicted_attendance + 6:.0f} people**."
        )
        
        with st.expander("See Raw Prediction Data"):
            st.write(input_data)
            st.write(prediction_output)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")