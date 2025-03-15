import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Streamlit App Title
st.title("📊 MAE & MFE Trading Dashboard")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

    # Ensure required columns exist
    required_columns = ["Duration", "MAE", "MFE"]
    if not all(col in df.columns for col in required_columns):
        st.error("Error: Required columns ('Duration', 'MAE', 'MFE') not found in the uploaded CSV.")
    else:
        # Extract Day of the Week from Column C and ensure it's formatted correctly
        df.rename(columns={df.columns[2]: "DayOfWeek"}, inplace=True)
       
