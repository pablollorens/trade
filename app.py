import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import hashlib
import math
import heapq
import time
from datetime import datetime, time as dtime
from typing import List, Dict, Any, Optional
from sklearn.linear_model import LinearRegression
import concurrent.futures  # For parallel processing

# =============================================================================
# UTILITY FUNCTION: RISK CALCULATION
# =============================================================================
def compute_risk_of_ruin(win_frac: np.ndarray, reward_to_risk: np.ndarray) -> np.ndarray:
    """
    Computes the risk of ruin based on win fraction and reward-to-risk.
    Uses a Kelly-derived formula.
    """
    kelly = np.where(reward_to_risk == 0, 0, win_frac - ((1 - win_frac) / reward_to_risk))
    risk = np.where(kelly > 0, np.exp(-5 * kelly), 1.0)
    return risk

# =============================================================================
# CONFIGURATION
# =============================================================================
RISK_PROFILES: Dict[str, Dict[str, float]] = {
    "FTMO": {"daily_multiplier": 2, "total_multiplier": 10},
    "Personal": {"daily_multiplier": 3, "total_multiplier": 15},
}

# =============================================================================
# DATA LAYER MODULE
# =============================================================================
class DataLayer:
    """Handles data loading, validation, and filtering."""
    def __init__(self, filepath: Optional[str] = None) -> None:
        if filepath:
            self.df = pd.read_csv(filepath)
        else:
            self.df = pd.read_csv("sample_data.csv")
        errors = self.validate_csv(self.df)
        if errors:
            raise ValueError(f"CSV validation errors: {', '.join(errors)}")
        if "StrategyID" not in self.df.columns:
            self.df["StrategyID"] = self.df.index.astype(str)
    
    @staticmethod
    def validate_csv(df: pd.DataFrame) -> List[str]:
        """
        Validates that required columns exist and have correct types.
        For "strike_rate", any numeric type is allowed.
        """
        required_columns: Dict[str, str] = {
            "asset": "object",
            "range_start": "object",
            "day_of_week": "object",
            "reward_to_risk": "float64"
        }
        errors = []
        for col in ["asset", "range_start", "day_of_week", "strike_rate", "reward_to_risk"]:
            if col not in df.columns:
                errors.append(f"Missing column: {col}")
        for col, dtype in required_columns.items():
            if col in df.columns and not np.issubdtype(df[col].dtype, np.dtype(dtype)):
                errors.append(f"{col} should be {dtype}")
        if "strike_rate" in df.columns:
            if not np.issubdtype(df["strike_rate"].dtype, np.number):
                errors.append("strike_rate should be numeric")
            elif (df["strike_rate"] < 0).any() or (df["strike_rate"] > 100).any():
                errors.append("Strike rates must be between 0 and 100%")
        return errors

    def filter_data(self, time_filter: bool, start_time: dtime, end_time: dtime, assets: List[str]) -> pd.DataFrame:
        """Filters the dataset by time and asset selection."""
        df_filtered = self.df.copy()
        df_filtered = df_filtered[df_filtered["asset"].isin(assets)]
        if time_filter:
            try:
                df_filtered["range_time"] = pd.to_datetime(df_filtered["range_start"], errors="coerce").dt.time
                df_filtered = df_filtered[df_filtered["range_time"].between(start_time, end_time)]
            except Exception as e:
                st.error(f"Error applying time filter: {e}")
        return df_filtered

    def compute_metrics(self, risk_pct: float) -> pd.DataFrame:
        """
        Computes key metrics for each strategy.
        Composite Score = Expected Value (%) × (1 – Risk of Ruin)
        """
        df = self.df.copy()
        df['WinFrac'] = df['strike_rate'] / 100
        df['EV_percent'] = (df['WinFrac'] * df['reward_to_risk'] - (1 - df['WinFrac'])) * risk_pct
        df['RiskOfRuin'] = compute_risk_of_ruin(df['WinFrac'], df['reward_to_risk'])
        df["CompositeScore"] = df["EV_percent"] * (1 - df["RiskOfRuin"])
        return df

# =============================================================================
# RISK ENGINE MODULE
# =============================================================================
class RiskEngineModule:
    """Provides risk calculations with configurable risk profiles."""
    def __init__(self, profile: str = "FTMO") -> None:
        self.profile = profile

    def add_drawdown_columns(self, df: pd.DataFrame, risk_fraction: float) -> pd.DataFrame:
        """Adds drawdown columns based on the selected risk profile."""
        multipliers = RISK_PROFILES.get(self.profile, RISK_PROFILES["FTMO"])
        df["WorstDailyDD"] = risk_fraction * multipliers["daily_multiplier"]
        df["WorstTotalDD"] = risk_fraction * multipliers["total_multiplier"]
        return df

    def calculate_drawdown_limits(self, risk_per_trade: float) -> (float, float):
        """Calculates daily and total drawdown limits."""
        if self.profile == "FTMO":
            return min(risk_per_trade * 2, 0.05), min(risk_per_trade * 10, 0.10)
        elif self.profile == "Aggressive":
            return risk_per_trade * 3, risk_per_trade * 15
        else:
            return risk_per_trade * 2.5, risk_per_trade * 12

# =============================================================================
# PORTFOLIO OPTIMIZER MODULE
# =============================================================================
class PortfolioOptimizerModule:
    """Evaluates strategy combinations and caches results."""
    def __init__(self, strategies: pd.DataFrame) -> None:
        self.strategies = strategies
        self.cache: Dict[str, float] = {}

    def _hash_combo(self, combo: tuple) -> str:
        """Generates a unique hash for the strategy combo."""
        return hashlib.md5(str(sorted(combo)).encode()).hexdigest()

    def evaluate_combo(self, combo: tuple, max_trades_per_day: int, diversity_alpha: float = 0.1,
                       min_assets: int = 0, min_days: int = 2) -> float:
        """
        Evaluates a combination of strategies.
        Returns the composite score or -∞ if constraints are not met.
        Requires at least 'min_days' distinct trading days.
        """
        combo_id = self._hash_combo(combo)
        if combo_id in self.cache:
            return self.cache[combo_id]
        subset = self.strategies[self.strategies['StrategyID'].isin(combo)]
        if subset['day_of_week'].nunique() < min_days:
            self.cache[combo_id] = -np.inf
            return -np.inf
        if (subset['day_of_week'].value_counts() > max_trades_per_day).any():
            self.cache[combo_id] = -np.inf
            return -np.inf
        if subset['asset'].nunique() < min_assets:
            self.cache[combo_id] = -np.inf
            return -np.inf
        avg_composite = subset["CompositeScore"].mean()
        rr_diversity = subset['reward_to_risk'].max() - subset['reward_to_risk'].min()
        final_score = avg_composite * (1 + diversity_alpha * rr_diversity)
        self.cache[combo_id] = final_score
        return final_score

# =============================================================================
# VISUALIZATION MODULE
# =============================================================================
class Visualization:
    """Provides helper methods for visualizations."""
    @staticmethod
    def rename_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            "asset": "Asset",
            "range_start": "Trading Range Start",
            "day_of_week": "Trading Day",
            "strike_rate": "Strike Rate (%)",
            "reward_to_risk": "Reward-to-Risk Ratio",
            "EV_percent": "Expected Value (%)",
            "WorstDailyDD": "Worst Daily Drawdown (%)",
            "WorstTotalDD": "Worst Total Drawdown (%)",
            "RiskOfRuin": "Risk of Ruin",
            "tp_percent": "Take Profit (%)",
            "sl_percent": "Stop Loss (%)",
            "CompositeScore": "Composite Score",
            "duration": "Avg Duration (min)"
        }
        return df.rename(columns=mapping)

    @staticmethod
    def weekly_range_table(df: pd.DataFrame, top_n: int) -> Dict[str, Any]:
        """
        Groups strategies by weekday and returns styled tables with top_n rows sorted by Composite Score.
        The table includes the average duration of trades if present.
        """
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=weekday_order, ordered=True)
        df = df.sort_values("day_of_week")
        if "duration" in df.columns:
            display_columns = ["asset", "range_start", "day_of_week", "strike_rate", "reward_to_risk",
                               "EV_percent", "RiskOfRuin", "tp_percent", "sl_percent", "duration", "CompositeScore"]
        else:
            display_columns = ["asset", "range_start", "day_of_week", "strike_rate", "reward_to_risk",
                               "EV_percent", "RiskOfRuin", "tp_percent", "sl_percent", "CompositeScore"]
        df["CompositeScore"] = df["EV_percent"] * (1 - df["RiskOfRuin"])
        groups: Dict[str, Any] = {}
        for day in weekday_order:
            day_df = df[df["day_of_week"] == day]
            if not day_df.empty:
                top_df = day_df.sort_values("CompositeScore", ascending=False).head(top_n)
                styled_table = Visualization.rename_columns_for_display(top_df[display_columns]).style \
                    .format({
                        "Reward-to-Risk Ratio": "{:.2f}",
                        "Take Profit (%)": "{:.2f}",
                        "Stop Loss (%)": "{:.2f}",
                        "Avg Duration (min)": "{:.2f}"
                    }) \
                    .background_gradient(cmap="RdYlGn", subset=["Expected Value (%)"]) \
                    .background_gradient(cmap="RdYlGn_r", subset=["Risk of Ruin"]) \
                    .background_gradient(cmap="PuBu", subset=["Composite Score"])
                groups[day] = styled_table
        return groups

    @staticmethod
    def ftmo_gauge(optimal_score: float) -> go.Figure:
        """
        Generates a gauge chart for FTMO account scoring guidelines.
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=optimal_score,
            title={'text': "Optimal Composite Score"},
            gauge={
                'axis': {'range': [0, 15]},
                'steps': [
                    {'range': [0, 5], 'color': "red"},
                    {'range': [5, 10], 'color': "yellow"},
                    {'range': [10, 15], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': optimal_score
                }
            }
        ))
        return fig

# =============================================================================
# EXPORT FUNCTION FOR WEEKLY RANGE ANALYSIS
# =============================================================================
def export_weekly_range_analysis(df: pd.DataFrame, top_n: int) -> str:
    """
    Combines the weekly range analysis for all weekdays into one DataFrame
    and returns a CSV string.
    """
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    analysis_frames = []
    for day in weekday_order:
        day_df = df[df["day_of_week"] == day].sort_values("CompositeScore", ascending=False).head(top_n)
        if not day_df.empty:
            day_df = day_df.copy()
            day_df["Weekday"] = day
            analysis_frames.append(day_df)
    if analysis_frames:
        export_df = pd.concat(analysis_frames, ignore_index=True)
    else:
        export_df = pd.DataFrame()
    return export_df.to_csv(index=False)

# =============================================================================
# AUTO BACKTESTER MODULE
# =============================================================================
class AutoBacktester:
    """
    Performs a grid search over parameters for portfolio optimization.

    This backtester uses its own parameter grid—ignoring the advanced settings—to vary:
      - risk_pct: scales EV_percent.
      - max_rr: upper bound for reward-to-risk.
      - max_trades: maximum trades per day allowed.
      - combo_size: number of strategies in a combo.
      - min_days: minimum distinct trading days required.
    """
    def __init__(self, strategies: pd.DataFrame, custom_grid: Optional[Dict[str, List[Any]]] = None) -> None:
        self.strategies = strategies
        if custom_grid is not None:
            self.param_grid = custom_grid
        else:
            self.param_grid = {
                'risk_pct': [0.5, 1.0, 1.5, 2.0],
                'max_rr': [1.0, 2.0, 3.0, 4.0, 5.0],
                'max_trades': [1, 2, 3, 4],
                'combo_size': [2, 3, 4, 5],
                'min_days': [2, 3, 4]
            }
    
    def _param_combinations(self) -> List[Dict[str, Any]]:
        from itertools import product
        keys = list(self.param_grid.keys())
        return [dict(zip(keys, values)) for values in product(*self.param_grid.values())]
    
    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        df_copy = self.strategies.copy()
        # Adjust EV_percent based on risk_pct parameter
        df_copy['EV_percent'] = df_copy['EV_percent'] * params['risk_pct']
        df_copy["CompositeScore"] = df_copy["EV_percent"] * (1 - df_copy["RiskOfRuin"])
        optimizer = PortfolioOptimizerModule(df_copy)
        candidates = (df_copy.groupby("day_of_week", group_keys=False)
                      .apply(lambda g: g.sort_values("CompositeScore", ascending=False).head(5))
                     ).reset_index(drop=True)
        unique_strategies = candidates["StrategyID"].unique()
        best_score = -np.inf
        for combo in combinations(unique_strategies, params['combo_size']):
            score = optimizer.evaluate_combo(combo, params['max_trades'], diversity_alpha=0.1, min_assets=0, min_days=params['min_days'])
            if score > best_score:
                best_score = score
        return best_score
    
    def run_optimization(self) -> pd.DataFrame:
        combinations_list = self._param_combinations()
        total = len(combinations_list)
        results = []
        start_time = time.time()
        progress_bar = st.progress(0)
        time_placeholder = st.empty()
        # Parallel processing with ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(self._evaluate_params, params): params for params in combinations_list}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                params = futures[future]
                try:
                    score = future.result()
                except Exception as e:
                    score = -np.inf
                results.append({**params, 'score': score})
                completed += 1
                progress_bar.progress(completed / total)
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = total - completed
                est_time_left = remaining * avg_time
                time_placeholder.write(f"Estimated time left: {est_time_left:.1f} seconds")
        return pd.DataFrame(results).sort_values('score', ascending=False)

def show_backtest_results(results_df: pd.DataFrame) -> None:
    cols = ["risk_pct", "max_rr", "max_trades", "combo_size", "min_days", "score"]
    styled = results_df[cols].style.format({'score': '{:.2f}'}).background_gradient(subset=['score'], cmap='RdYlGn')
    st.markdown(styled.to_html(), unsafe_allow_html=True)

# =============================================================================
# FTMO SAFETY CHECK
# =============================================================================
def ftmo_safety_check(portfolio: pd.DataFrame) -> None:
    """
    Checks if the portfolio exceeds FTMO risk limits.
    """
    dd_daily = portfolio['WorstDailyDD'].sum()
    dd_total = portfolio['WorstTotalDD'].sum()
    if dd_daily > 0.05 or dd_total > 0.10:
        st.error("⚠️ FTMO Rules At Risk! Adjust parameters.")
        st.progress(min(dd_daily / 0.05, 1.0))
    else:
        st.success("Portfolio is within FTMO risk limits.")

# =============================================================================
# CORRELATION ANALYSIS & RECOMMENDATIONS
# =============================================================================
def run_correlation_analysis(df: pd.DataFrame) -> None:
    """
    Computes and displays a correlation heatmap for key numeric metrics.
    """
    numeric_cols = []
    for col in ["EV_percent", "reward_to_risk", "RiskOfRuin", "CompositeScore", "WorstDailyDD", "WorstTotalDD"]:
        if col in df.columns:
            numeric_cols.append(col)
    if numeric_cols:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Not enough numeric data for correlation analysis.")

def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates recommendations by selecting the top row (highest Composite Score)
    for each weekday.
    """
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    recommendations = []
    for day in weekday_order:
        day_df = df[df["day_of_week"] == day].copy()
        if not day_df.empty:
            day_df = day_df.sort_values("CompositeScore", ascending=False)
            recommendations.append(day_df.iloc[0])
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        return rec_df
    else:
        return pd.DataFrame()

# =============================================================================
# MAIN APP
# =============================================================================
def main() -> None:
    st.set_page_config(layout="wide")
    
    st.title("🛡️ Ultimate FTMO/Personal Portfolio Builder")
    st.markdown("""
    This dashboard builds an optimal trading portfolio while ensuring adherence to FTMO risk rules.
    """)
    
    # Automatically load default CSV ("sample_data.csv")
    data_layer = DataLayer("sample_data.csv")
    st.info("Using 'sample_data.csv' by default.")
    
    raw_df = data_layer.df.copy()
    if "StrategyID" not in data_layer.df.columns:
        data_layer.df["StrategyID"] = data_layer.df.index.astype(str)
        raw_df["StrategyID"] = raw_df.index.astype(str)
    
    with st.sidebar.expander("Data Filters"):
        st.markdown("Filter by time and asset.")
        apply_time_filter = st.checkbox("Apply Time Filter", value=False, help="Toggle to filter by trading start time.")
        if apply_time_filter:
            start_time_input = st.time_input("Start Time", value=dtime(0, 0), help="Start (24‑hr).")
            end_time_input = st.time_input("End Time", value=dtime(23, 59), help="End (24‑hr).")
        all_assets = sorted(raw_df["asset"].unique())
        selected_assets = st.multiselect("Select Assets", options=all_assets, default=all_assets, help="Choose assets.")
    
    st.sidebar.header("Basic Settings")
    risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1, help="Risk per trade.")
    risk_profile = st.sidebar.radio("Risk Profile", ["FTMO", "Personal", "Aggressive"], index=0, help="Risk profile.")
    risk_engine = RiskEngineModule(risk_profile)
    daily_dd, total_dd = risk_engine.calculate_drawdown_limits(risk_pct / 100)
    
    with st.sidebar.expander("Advanced Settings"):
        min_rr = st.slider("Minimum Reward-to-Risk", 0.5, 5.0, 1.0, 0.1, help="Minimum Reward-to-Risk.")
        max_rr = st.slider("Maximum Reward-to-Risk", 0.5, 10.0, 3.0, 0.1, help="Maximum Reward-to-Risk.")
        top_ranges = st.selectbox("Top ranges per weekday", [5, 10, 15, 20, 25, 30], index=0, help="Number of top ranges to display per weekday.")
        max_trades_per_day = st.slider("Max Trades per Day in Combo", 1, 5, 1, help="Max trades per day.")
        combo_size = st.slider("Number of Strategies in Combo", 2, 5, 3, help="Strategies per combo.")
        diversity_alpha = st.slider("Reward-to-Risk Diversity Factor", 0.0, 0.5, 0.1, 0.05, help="Diversity bonus factor.")
        min_assets = st.slider("Minimum Distinct Assets", 0, 10, 1, help="Minimum distinct assets in a combo.")
        trades_per_week = st.slider("Average Trades per Week", 1, 20, 5, help="Estimated trades per week.")
    
    st.sidebar.header("Backtesting Settings")
    min_risk_bt = st.sidebar.number_input("Min Risk per Trade (%) for Backtesting", value=0.5, step=0.1)
    max_risk_bt = st.sidebar.number_input("Max Risk per Trade (%) for Backtesting", value=1.0, step=0.1)
    num_risk_steps = st.sidebar.number_input("Number of Risk Steps", value=3, step=1)
    risk_values = list(np.linspace(min_risk_bt, max_risk_bt, int(num_risk_steps)))
    custom_grid = {
        'risk_pct': risk_values,
        'max_rr': [1.0, 2.0, 3.0, 4.0, 5.0],
        'max_trades': [1, 2, 3, 4],
        'combo_size': [2, 3, 4, 5],
        'min_days': [2, 3, 4]
    }
    
    st.sidebar.header("Parameter Optimization")
    # Backtesting will run when the button in the Backtesting tab is pressed.
    
    st.sidebar.header("Export Data")
    export_csv = st.sidebar.button("Export Weekly Range Analysis CSV")
    
    # Compute metrics and filter data for main app
    data_layer.df = data_layer.compute_metrics(risk_pct)
    df_filtered = data_layer.filter_data(apply_time_filter,
                                         start_time_input if apply_time_filter else dtime(0, 0),
                                         end_time_input if apply_time_filter else dtime(23, 59),
                                         selected_assets)
    df_filtered = risk_engine.add_drawdown_columns(df_filtered, risk_pct/100)
    mask = (df_filtered['WorstDailyDD'] <= daily_dd) & (df_filtered['WorstTotalDD'] <= total_dd) & \
           (df_filtered['reward_to_risk'] >= 1.0) & (df_filtered['reward_to_risk'] <= max_rr)
    df_filtered = df_filtered[mask].copy()
    if df_filtered.empty:
        st.error("No strategies meet the specified risk parameters. Adjust your filters.")
    
    # Export weekly range analysis if requested
    if export_csv:
        csv_data = export_weekly_range_analysis(df_filtered, top_ranges)
        st.download_button(label="Download Weekly Range Analysis CSV",
                           data=csv_data,
                           file_name="weekly_range_analysis.csv",
                           mime="text/csv")
    
    tabs = st.tabs(["Dashboard", "Backtesting", "Data Analysis"])
    
    with tabs[0]:
        st.header("Dashboard")
        st.subheader("Weekly Range Analysis")
        groups = Visualization.weekly_range_table(df_filtered, top_ranges)
        if not groups:
            st.write("No top ranges found. Check filters or Trading Day values.")
        else:
            for day, styled_table in groups.items():
                st.markdown(f"**{day}:**")
                st.markdown(styled_table.to_html(), unsafe_allow_html=True)
        
        st.subheader("Portfolio Optimization")
        df_filtered["CompositeScore"] = df_filtered["EV_percent"] * (1 - df_filtered["RiskOfRuin"])
        candidates = (df_filtered.groupby("day_of_week", group_keys=False)
                      .apply(lambda g: g.sort_values("CompositeScore", ascending=False).head(top_ranges))
                     ).reset_index(drop=True)
        unique_strategies = candidates["StrategyID"].unique()
        optimizer = PortfolioOptimizerModule(df_filtered)
        st.info("Optimizing strategy combos...")
        top_combos_heap = []
        total_combos = math.comb(len(unique_strategies), combo_size)
        progress_bar = st.progress(0)
        for i, combo in enumerate(combinations(unique_strategies, combo_size)):
            progress_bar.progress((i+1)/total_combos)
            score = optimizer.evaluate_combo(combo, max_trades_per_day, diversity_alpha, min_assets, min_days=2)
            if score == -np.inf:
                continue
            subset = df_filtered[df_filtered["StrategyID"].isin(combo)]
            weekly_profit = subset["CompositeScore"].mean() * trades_per_week
            details = "\n".join([
                f"Asset: {row['asset']} | Range: {row['range_start']} | TP: {row.get('tp_percent', 'N/A')} | SL: {row.get('sl_percent', 'N/A')} | RR: {row['reward_to_risk']:.2f} | Comp: {row['CompositeScore']:.1f}"
                for idx, row in subset.iterrows()
            ])
            details += f"\nExpected Weekly Profit Gain (%): {weekly_profit:.2f}"
            if len(top_combos_heap) < 5:
                heapq.heappush(top_combos_heap, (score, combo, details))
            else:
                if score > top_combos_heap[0][0]:
                    heapq.heapreplace(top_combos_heap, (score, combo, details))
        if top_combos_heap:
            best_combo = max(top_combos_heap, key=lambda x: x[0])[1]
            st.subheader("🏆 Optimal Strategy Combo")
            st.write(df_filtered[df_filtered["StrategyID"].isin(best_combo)])
            st.subheader("Top 5 Strategy Combos")
            combos_df = pd.DataFrame(top_combos_heap, columns=["Combo Score", "Strategy Combo", "Strategy Details"])
            st.dataframe(combos_df)
        else:
            st.error("No valid strategy combos found.")
        
        st.subheader("FTMO Account Scoring Guidelines")
        if top_combos_heap:
            best_score_value = max(top_combos_heap, key=lambda x: x[0])[0]
            fig_gauge = Visualization.ftmo_gauge(best_score_value)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("""
            **Guidelines:**
            - Red (0–5): Not ideal.
            - Yellow (5–10): Acceptable.
            - Green (>10): Ideal.
            
            **Risk Limits:**
            - Daily Loss: 2–3%
            - Total Drawdown: <10%
            """)
            ftmo_safety_check(df_filtered)
        else:
            st.write("No optimal composite score available yet.")
    
    with tabs[1]:
        st.header("Backtesting")
        if st.button("Start Backtesting"):
            backtester = AutoBacktester(df_filtered, custom_grid=custom_grid)
            time_placeholder = st.empty()
            results = backtester.run_optimization()
            show_backtest_results(results)
    
    with tabs[2]:
        st.header("Data Analysis")
        st.subheader("Correlation Analysis")
        run_correlation_analysis(df_filtered)
        
        st.subheader("Recommendations")
        rec_df = generate_recommendations(df_filtered)
        if rec_df.empty:
            st.write("No recommendations available. Try adjusting your filters.")
        else:
            st.dataframe(Visualization.rename_columns_for_display(rec_df))
            st.markdown("""
            **Rationale:**
            - The recommendations select the top range (highest Composite Score) for each weekday.
            - This ensures diversification across the week.
            - The selected ranges have favorable Reward-to-Risk ratios and controlled Risk of Ruin, making them safe for FTMO.
            - TP and SL values are chosen to capture breakout moves while limiting losses.
            """)
            
def run_correlation_analysis(df: pd.DataFrame) -> None:
    """
    Computes and displays a correlation heatmap for key numeric metrics.
    """
    numeric_cols = []
    for col in ["EV_percent", "reward_to_risk", "RiskOfRuin", "CompositeScore", "WorstDailyDD", "WorstTotalDD"]:
        if col in df.columns:
            numeric_cols.append(col)
    if numeric_cols:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Not enough numeric data for correlation analysis.")

def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates recommendations by selecting the top row (highest Composite Score)
    for each weekday.
    """
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    recommendations = []
    for day in weekday_order:
        day_df = df[df["day_of_week"] == day].copy()
        if not day_df.empty:
            day_df = day_df.sort_values("CompositeScore", ascending=False)
            recommendations.append(day_df.iloc[0])
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        return rec_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    main()