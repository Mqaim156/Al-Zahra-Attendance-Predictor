import streamlit as st
import pandas as pd
from pathlib import Path

# We import the prediction function from the file we already created
try:
    from model_pipeline import predict_pipeline
except ImportError:
    st.error("Error: 'model_pipeline.py' not found. Please make sure it's in the same folder as app.py.")
    st.stop()

# --- 1. SET UP FILE PATH (RELATIVE PATH) ---
logo_path = "az-logo.png"

# --- 2. App Configuration ---
# This must be the first Streamlit command.
st.set_page_config(
    page_title="Al-Zahra Attendance Predictor",
    page_icon=logo_path,
    layout="wide",  # Use "wide" layout for a more professional look
    initial_sidebar_state="expanded"
)

# --- 3. CUSTOM CSS TO INJECT ---
# This CSS will add "cards", round corners, and a new header color.
def load_css():
    st.markdown("""
        <style>
            /* Main container style */
            .main .block-container {
                padding-top: 2rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }

            /* Create a "card" class for containers */
            .card {
                background-color: #FFFFFF;
                border-radius: 10px;
                padding: 25px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }

            /* Style for headers */
            h1, h2 {
                color: #2a3f5f; /* A professional, deep blue */
            }
            h3 {
                color: #334e7c;
            }

            /* Style the main predict button */
            .stButton>button {
                background-color: #2a3f5f;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
                border: none;
                width: 100%;
            }
            .stButton>button:hover {
                background-color: #334e7c;
                color: white;
                border: none;
            }
            
            /* Clean up sidebar */
            .st-emotion-cache-16txtl3 {
                padding-top: 1rem;
            }
            
            /* Style the metric */
            .st-emotion-cache-1b0udgb {
                background-color: #f9f9f9;
                border-radius: 10px;
                padding: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

# --- 4. PASSWORD PROTECTION ---
def check_password():
    """Returns True if the user has entered the correct password."""
    if "APP_PASSWORD" not in st.secrets:
        st.error("Password not set. Please contact the administrator.")
        st.stop()
    if st.session_state.get("logged_in", False):
        return True

    # Center the login form
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.container(border=True, height=250):
            st.image(logo_path, width=100)
            st.header("Login")
            password = st.text_input("Enter Password:", type="password", key="password_input")
            
            if st.button("Login", use_container_width=True, type="primary"):
                if password == st.secrets["APP_PASSWORD"]:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
    return False

# --- 5. MAIN APP LOGIC ---
def run_main_app():
    """This function runs the main app after password is verified."""
    
    # Load the custom CSS
    load_css()
    
    # --- App Header ---
    col1, col2 = st.columns([1, 4]) 
    with col1:
        st.image(logo_path, width=100) 
    with col2:
        st.title("Al-Zahra Event Predictor")
        st.markdown("Enter event details below to predict the attendance.")

    # --- Input Section ---
    st.sidebar.image(logo_path)
    st.sidebar.header("Navigation")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
    
    st.subheader("Event Features")
    
    # Use st.form to group inputs
    with st.form(key="prediction_form"):
        # We'll use columns to arrange the inputs neatly
        col1, col2, col3 = st.columns(3)
        
        with col1:
            program_type_options = ["small", "medium", "large", "extra large"]
            program_type = st.selectbox(
                "Program Type:",
                options=program_type_options,
                help="Select the type of program."
            )
            
            is_food = st.toggle(
                "Is food being served?",
                value=True,
                help="Check this if food (e.g., dinner, Iftar) will be provided."
            )
        
        with col2:
            is_weekend = st.toggle(
                "Is it a weekend?",
                value=True,
                help="Check this if the event is on a Saturday or Sunday."
            )
            
            special_speaker_flag = st.toggle(
                "Is there a special speaker?",
                value=False,
                help="Check this if a well-known or guest speaker is featured."
            )
        
        with col3:
            is_summer = st.toggle(
                "Is it in summer?",
                value=False,
                help="Check this if the event is in June, July, or August."
            )
            
            is_special_month = st.toggle(
                "Is it a special Islamic month?",
                value=True,
                help="Check this for months like Muharram, Ramadan, etc."
            )
        
        # The "Submit" button for the form
        predict_button = st.form_submit_button(
            label="Predict Attendance", 
            type="primary", 
            use_container_width=True
        )

    # --- Prediction Logic & Output ---
    if predict_button:
        # Collect inputs
        input_data = {
            "is_summer": 1 if is_summer else 0,
            "is_weekend": 1 if is_weekend else 0,
            "program_type": program_type,
            "special_speaker_flag": 1 if special_speaker_flag else 0,
            "is_special_month": 1 if is_special_month else 0,
            "is_food": 1 if is_food else 0
        }

        # Call the prediction pipeline
        try:
            prediction_output = predict_pipeline(input_data)
            predicted_attendance = prediction_output.get("predicted_attendance")

            # Display the result in its own "card"
            st.subheader("📈 Predicted Attendance")
            with st.container(border=True):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric(
                        label="Predicted Guests",
                        value=f"{predicted_attendance:.0f}"
                    )
                
                with col2:
                    st.info(
                        f"**Confidence Range:** Based on the model's R² score of 0.68 and an average error (MAE) of ~6, you can "
                        f"confidently plan for an attendance between **{predicted_attendance - 6:.0f} and {predicted_attendance + 6:.0f} people**."
                    )
                
                with st.expander("See Raw Prediction Data"):
                    st.write(input_data)
                    st.write(prediction_output)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- Run the App ---
if not check_password():
    st.stop()  # Stop the app if password is wrong

run_main_app()  # Run the main app if password is correct