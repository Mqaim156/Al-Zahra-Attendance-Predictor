import streamlit as st
import pandas as pd

# --- 1. Check Login Status ---
# Check if user is logged in (from the session state in app.py)
if not st.session_state.get("logged_in", False):
    st.error("Please login first on the main page.")
    st.stop()

# --- 2. Custom CSS for Styling ---
# --- Page Content ---
st.title("📊 Historic Data Explorer")
st.markdown("Upload your original `attendance.csv` file to see charts and insights from past events.")

uploaded_file = st.file_uploader("Upload your attendance.csv file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- Show a sample of the data ---
        with st.expander("Show Raw Data Sample"):
            st.dataframe(df.head())

        # --- Use tabs for a cleaner layout ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "Attendance Over Time", 
            "By Program Type", 
            "By Day", 
            "Impact of Food"
        ])

        with tab1:
            st.subheader("Attendance Over Time")
            st.line_chart(df['attendance'], use_container_width=True)
            st.caption("This chart plots attendance for each event, in chronological order.")

        with tab2:
            st.subheader("Average Attendance by Program Type")
            avg_by_program = df.groupby('program_type')['attendance'].mean().sort_values(ascending=False)
            st.bar_chart(avg_by_program, use_container_width=True)

        with tab3:
            st.subheader("Average Attendance by Weekend vs. Weekday")
            avg_by_weekend = df.groupby("is_weekend")['attendance'].mean()
            avg_by_weekend.index = avg_by_weekend.index.map({0: 'Weekday', 1: 'Weekend'})
            st.bar_chart(avg_by_weekend, use_container_width=True)
        
        with tab4:
            st.subheader("Impact of Food on Attendance")
            avg_by_food = df.groupby('is_food')['attendance'].mean()
            avg_by_food.index = avg_by_food.index.map({0: 'No Food', 1: 'Food Served'})
            st.bar_chart(avg_by_food, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        st.info("Please ensure your CSV file is formatted correctly.")

else:
    st.info("Please upload your `attendance.csv` file to get started.")