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

    # Ensure the required columns exist
    required_columns = ["Duration", "MAE", "MFE"]
    if not all(col in df.columns for col in required_columns):
        st.error("Error: Required columns ('Duration', 'MAE', 'MFE') not found in the uploaded CSV.")
    else:
        # Extract Day of the Week from Column C and ensure it's properly formatted
        df.rename(columns={df.columns[2]: "DayOfWeek"}, inplace=True)

        df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
        df['DayOfWeek'] = df['DayOfWeek'].astype(str)

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
            df['DayOfWeek'].unique().tolist(), 
            default=df['DayOfWeek'].unique().tolist()
        )

        # Trade duration filter (user selects min and max duration)
        min_duration, max_duration = st.sidebar.slider(
            "Select Trade Duration Range", 
            int(df["Duration"].min()), 
            int(df["Duration"].max()), 
            (int(df["Duration"].min()), int(df["Duration"].max()))
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
            (filtered_df["Duration"] >= min_duration) & (filtered_df["Duration"] <= max_duration)
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
