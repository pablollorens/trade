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
        df_1y = df[df['Datetime'] >= one_year_ago]
        df_6m = df[df['Datetime'] >= six_months_ago]
        df_3m = df[df['Datetime'] >= three_months_ago]

        # Apply day-of-the-week filter
        df_1y = df_1y[df_1y['DayOfWeek'].isin(days_selected)]
        df_6m = df_6m[df_6m['DayOfWeek'].isin(days_selected)]
        df_3m = df_3m[df_3m['DayOfWeek'].isin(days_selected)]

        # Apply trade duration filter
        df_1y = df_1y[(df_1y["Duration"] >= min_duration) & (df_1y["Duration"] <= max_duration)]
        df_6m = df_6m[(df_6m["Duration"] >= min_duration) & (df_6m["Duration"] <= max_duration)]
        df_3m = df_3m[(df_3m["Duration"] >= min_duration) & (df_3m["Duration"] <= max_duration)]

        # Define percentile groups
        mae_percentiles_values = [0.7, 0.8, 0.9]  # 70th, 80th, 90th for MAE
        mfe_percentiles_values = [0.3, 0.2, 0.1]  # 30th, 20th, 10th for MFE

        # Compute MAE and MFE percentiles
        mae_percentiles = {
            "1Yr": [df_1y['MAE'].quantile(p) for p in mae_percentiles_values],
            "6Mo": [df_6m['MAE'].quantile(p) for p in mae_percentiles_values],
            "3Mo": [df_3m['MAE'].quantile(p) for p in mae_percentiles_values],
        }

        mfe_percentiles = {
            "1Yr": [df_1y['MFE'].quantile(p) for p in mfe_percentiles_values],
            "6Mo": [df_6m['MFE'].quantile(p) for p in mfe_percentiles_values],
            "3Mo": [df_3m['MFE'].quantile(p) for p in mfe_percentiles_values],
        }

        # Compute median of each percentile across the 3 timeframes
        total_median_mae = [np.median([mae_percentiles["1Yr"][i], mae_percentiles["6Mo"][i], mae_percentiles["3Mo"][i]]) for i in range(3)]
        total_median_mfe = [np.median([mfe_percentiles["1Yr"][i], mfe_percentiles["6Mo"][i], mfe_percentiles["3Mo"][i]]) for i in range(3)]

        # Calculate Expected Value (EV) using the median MFE and MAE
        ev_values = [(total_median_mfe[i] - total_median_mae[i]) for i in range(3)]

        # Display Expected Value Table
        st.subheader("📊 Expected Value (EV) Based on Median MFE & MAE")
        ev_table = pd.DataFrame({
            "Percentile": ["EV (70th MAE, 30th MFE)", "EV (80th MAE, 20th MFE)", "EV (90th MAE, 10th MFE)"],
            "Expected Value": ev_values
        })
        st.dataframe(ev_table)

        # Separate Scatter Plots
        st.subheader("📈 MAE Scatter Plot")
        fig_mae = px.scatter(df_1y, x=df_1y.index, y="MAE", title="Scatter Plot of MAE",
                             labels={"MAE": "Maximum Adverse Excursion", "index": "Trade Index"})
        st.plotly_chart(fig_mae)

        st.subheader("📈 MFE Scatter Plot")
        fig_mfe = px.scatter(df_1y, x=df_1y.index, y="MFE", title="Scatter Plot of MFE",
                             labels={"MFE": "Maximum Favorable Excursion", "index": "Trade Index"})
        st.plotly_chart(fig_mfe)

else:
    st.info("Upload a CSV file to get started.")
