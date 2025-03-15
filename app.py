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
    df['DayOfWeek'] = df['Datetime'].dt.day_name()  # Extract day of the week

    # Ensure the trade duration column exists
    if 'TradeDuration' not in df.columns:
        st.error("Error: 'TradeDuration' column not found in the CSV.")
    else:
        df['TradeDuration'] = pd.to_numeric(df['TradeDuration'], errors='coerce')

        # Define timeframes based on the latest date in the dataset
        latest_date = df['Datetime'].max()
        one_year_ago = latest_date - pd.DateOffset(years=1)
        six_months_ago = latest_date - pd.DateOffset(months=6)
        three_months_ago = latest_date - pd.DateOffset(months=3)

        # Sidebar Filters
        st.sidebar.header("Filters")

        # User selects timeframe for analysis
        timeframe = st.sidebar.selectbox("Select Timeframe", ["1-Year", "6-Month", "3-Month"])

        # Day of the week selection filter
        days_selected = st.sidebar.multiselect(
            "Select Days of the Week", 
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], 
            default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        )

        # Trade duration filter (user selects min and max duration)
        min_duration, max_duration = st.sidebar.slider(
            "Select Trade Duration Range", 
            int(df["TradeDuration"].min()), 
            int(df["TradeDuration"].max()), 
            (int(df["TradeDuration"].min()), int(df["TradeDuration"].max()))
        )

        # Filter data based on timeframe selection
        if timeframe == "1-Year":
            filtered_df = df[df['Datetime'] >= one_year_ago]
        elif timeframe == "6-Month":
            filtered_df = df[df['Datetime'] >= six_months_ago]
        else:
            filtered_df = df[df['Datetime'] >= three_months_ago]

        # Apply day-of-the-week filter
        filtered_df = filtered_df[filtered_df['DayOfWeek'].isin(days_selected)]

        # Apply trade duration filter
        filtered_df = filtered_df[
            (filtered_df["TradeDuration"] >= min_duration) & (filtered_df["TradeDuration"] <= max_duration)
        ]

        # Extract MAE & MFE
        filtered_df = filtered_df[['MAE', 'MFE']].dropna()

        # Define new percentile settings
        mae_percentiles_values = [0.7, 0.8, 0.9]  # 70th, 80th, 90th for MAE
        mfe_percentiles_values = [0.3, 0.2, 0.1]  # 30th, 20th, 10th for MFE

        # Calculate percentiles
        percentile_data = {
            "Percentile": [f"{int(p * 100)}th" for p in mae_percentiles_values],
            "MAE": [filtered_df['MAE'].quantile(p) for p in mae_percentiles_values],
            "MFE": [filtered_df['MFE'].quantile(p) for p in mfe_percentiles_values]
        }

        # Convert to DataFrame and display as a table
        percentile_df = pd.DataFrame(percentile_data)
        st.subheader("📋 MAE & MFE Percentiles")
        st.dataframe(percentile_df)

        # Compute SL/TP based on median percentiles
        sl = np.median(percentile_df['MAE'])
        tp = np.median(percentile_df['MFE'])

        st.subheader("📌 Suggested SL/TP Levels")
        st.write(f"🔹 **Stop-Loss (SL):** {sl:.2f}")
        st.write(f"🔹 **Take-Profit (TP):** {tp:.2f}")

        # Scatter Plot Visualization
        st.subheader("📈 MAE vs. MFE Scatter Plot")
        fig = px.scatter(filtered_df, x="MAE", y="MFE", title="Scatter Plot of MAE vs MFE",
                        labels={"MAE": "Maximum Adverse Excursion", "MFE": "Maximum Favorable Excursion"})
        st.plotly_chart(fig)

else:
    st.info("Upload a CSV file to get started.")
