import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Streamlit App Title
st.title("📊 MAE & MFE Trading Dashboard")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Load CSV and display debug info
        df = pd.read_csv(uploaded_file)
        st.write("✅ File uploaded successfully!")
        st.write("📊 Data Preview:", df.head())  # Show first 5 rows
        st.write("🔍 Columns in file:", df.columns.tolist())

        # Convert Datetime column if it exists
        if "Datetime" in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        else:
            st.warning("⚠️ 'Datetime' column not found in CSV.")

        # Ensure required columns exist
        required_columns = ["Duration", "MAE", "MFE"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"⚠️ Missing required columns: {missing_columns}")
        else:
            # Extract Day of the Week from Column C (if exists)
            if len(df.columns) > 2:
                df.rename(columns={df.columns[2]: "DayOfWeek"}, inplace=True)
                df['DayOfWeek'] = df['DayOfWeek'].astype(str)

            df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

            # Show updated dataset preview
            st.write("✅ Processed Data Preview:", df.head())

            # =============================
            # 🔍 Expected Value (EV) Tester
            # =============================
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

    except Exception as e:
        st.error(f"⚠️ Error loading file: {e}")

else:
    st.info("Upload a CSV file to get started.")
