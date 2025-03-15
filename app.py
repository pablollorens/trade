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

        # ================================
        # 🔍 Expected Value (EV) Tester
        # ================================
        st.header("🔍 Expected Value (EV) Tester")

        # User inputs for MAE, MFE thresholds, and dollar amount per trade
        user_mae = st.number_input("Enter MAE Threshold (SL Level)", min_value=0.0, step=0.01, value=0.2)
        user_mfe = st.number_input("Enter MFE Threshold (TP Level)", min_value=0.0, step=0.01, value=0.5)
        trade_amount = st.number_input("Enter Dollar Amount per Trade ($)", min_value=1.0, step=1.0, value=100.0)

        # Day of the week selection filter
        days_selected = st.multiselect(
            "Filter by Days of the Week", 
            df['DayOfWeek'].unique().tolist(), 
            default=df['DayOfWeek'].unique().tolist()
        )

        # Filter dataset based on selected days
        df_filtered = df[df['DayOfWeek'].isin(days_selected)]

        # Count Wins (TP) and Losses (SL)
        win_trades = df_filtered[df_filtered["MFE"] >= user_mfe].shape[0]
        loss_trades = df_filtered[(df_filtered["MAE"] >= user_mae) | (df_filtered["MFE"] < user_mfe)].shape[0]
        total_trades = win_trades + loss_trades

        if total_trades > 0:
            win_rate = win_trades / total_trades
            loss_rate = loss_trades / total_trades

            # Calculate Expected Value (EV)
            expected_value = (win_rate * trade_amount) - (loss_rate * trade_amount)

            # Display Results
            st.subheader("📊 EV Tester Results")
            st.write(f"✔️ **Win Rate:** {win_rate:.2%}")
            st.write(f"❌ **Loss Rate:** {loss_rate:.2%}")
            st.write(f"💰 **Expected Value per Trade:** ${expected_value:.2f}")

        else:
            st.warning("No trades found that match the selected criteria. Adjust your inputs.")

        # =====================================
        # 📈 MAE & MFE Percentile Analysis
        # =====================================
        st.header("📈 MAE & MFE Analysis")
        st.info("Continue analyzing percentile tables, scatter plots, and further insights below.")

        # Define timeframes based on the latest date in the dataset
        latest_date = df['Datetime'].max()
        one_year_ago = latest_date - pd.DateOffset(years=1)
        six_months_ago = latest_date - pd.DateOffset(months=6)
        three_months_ago = latest_date - pd.DateOffset(months=3)

        # Filter data based on timeframe selection
        df_1y = df[df['Datetime'] >= one_year_ago]
        df_6m = df[df['Datetime'] >= six_months_ago]
        df_3m = df[df['Datetime'] >= three_months_ago]

        # Apply day-of-the-week filter
        df_1y = df_1y[df_1y['DayOfWeek'].isin(days_selected)]
        df_6m = df_6m[df_6m['DayOfWeek'].isin(days_selected)]
        df_3m = df_3m[df_3m['DayOfWeek'].isin(days_selected)]

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
