import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Historic Data", page_icon="📊")
st.title("📊 Historic Attendance Data Explorer")
st.markdown("Upload your original 'attendance.csv' file to see chart visualizations of historic data.")

# File uploader
uploaded_file = st.file_uploader("Upload your attendance.csv", type="csv")

if uploaded_file is not None:
    try:
        # Load the data
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        # Show a sample of the data
        st.subheader("Sample of Uploaded Data")
        st.dataframe(df.head())

        # Basic visualizations
        # Chart 1: Attendance Over Time
        st.subheader("Attendance Over Time")
        
        st.line_chart(df["attendance"], use_container_width=True)
        st.caption("This char plots attendance for each event, in chronological order.")

        # Chart 2: Average Attendance by Program Type
        st.subheader("Average Attendance by Program Type")
        avg_by_program = df.groupby("program_type")["attendance"].mean().sort_values()
        st.bar_chart(avg_by_program, use_container_width=True)

        # Chart 3: Attendance Distribution
        st.subheader("Average Attendance by Day of Week")
        avg_by_weekend = df.groupby("is_weekend")['attendance'].mean()
        avg_by_weekend.index = avg_by_weekend.index.map({0: 'Weekday', 1: 'Weekend'})
        st.bar_chart(avg_by_weekend, use_container_width=True)

        #Chart 4: Attendance Food vs No Food
        st.subheader("Average Attendance: Food vs No Food")
        avg_by_food = df.groupby("is_food")['attendance'].mean()
        avg_by_food.index = ['No Food', 'Food']
        st.bar_chart(avg_by_food, use_container_width=True)

    except Exception as e:
        st.error(f"Error Processing CSV file: {e}")
        st.info("Please ensure your CSV file is formatted correctly")


else:
    st.info("Please upload your 'attendance.csv' file to get started")

