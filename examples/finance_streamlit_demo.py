#!/usr/bin/env python
"""
Streamlit demo for SEC Filing Analyzer.

This is a sleek, lean demo that showcases the core functionality of the SEC Filing Analyzer.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import project modules
# Import the DuckDB store extension
# This import adds methods to the OptimizedDuckDBStore class
from examples import duckdb_store_extension  # noqa
from sec_filing_analyzer.quantitative.storage import OptimizedDuckDBStore

# Set page config
st.set_page_config(
    page_title="SEC Filing Analyzer Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #FAFAFA;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "etl_complete" not in st.session_state:
    st.session_state.etl_complete = False
if "selected_company" not in st.session_state:
    st.session_state.selected_company = None
if "years" not in st.session_state:
    st.session_state.years = []
if "metrics" not in st.session_state:
    st.session_state.metrics = []
if "db_store" not in st.session_state:
    # Initialize DuckDB store
    db_path = "data/db_backup/financial_data.duckdb"
    st.session_state.db_store = OptimizedDuckDBStore(db_path=db_path)


# Function to run ETL process
def run_etl(ticker, years, test_mode=False):
    """Run the ETL process for the selected company and years."""
    st.session_state.etl_complete = False
    st.session_state.selected_company = ticker
    st.session_state.years = years

    # Build the command
    cmd = ["python", str(Path(project_root) / "examples" / "run_nvda_etl.py"), "--ticker", ticker]

    # Add years
    for year in years:
        cmd.extend(["--years", str(year)])

    # Add test mode if needed
    if test_mode:
        cmd.append("--test-mode")

    # Run the command
    with st.spinner(f"Running ETL process for {ticker}..."):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Show output in real-time
        output_placeholder = st.empty()
        output = ""

        while True:
            output_line = process.stdout.readline()
            if output_line == "" and process.poll() is not None:
                break
            if output_line:
                output += output_line
                output_placeholder.code(output)

        # Get return code
        return_code = process.poll()

        if return_code == 0:
            st.session_state.etl_complete = True
            st.success(f"ETL process for {ticker} completed successfully!")
        else:
            st.error(f"ETL process for {ticker} failed with return code {return_code}")
            st.code(process.stderr.read())


# Function to query financial data
def query_financial_data(ticker, metrics, start_year, end_year):
    """Query financial data for the selected company, metrics, and years."""
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    results = st.session_state.db_store.query_financial_facts(
        ticker=ticker, metrics=metrics, start_date=start_date, end_date=end_date
    )

    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        # Convert period_end_date to datetime
        df["period_end_date"] = pd.to_datetime(df["period_end_date"])
        # Sort by period_end_date
        df = df.sort_values("period_end_date")
        return df
    else:
        return pd.DataFrame()


# Function to query revenue
def query_revenue(ticker, year):
    """Query revenue for the selected company and year."""
    cmd = ["python", str(Path(project_root) / "examples" / "query_revenue.py"), "--ticker", ticker, "--year", str(year)]

    # Run the command
    with st.spinner(f"Querying revenue for {ticker} in {year}..."):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Get output
        stdout, stderr = process.communicate()

        # Get return code
        return_code = process.poll()

        if return_code == 0:
            return stdout.strip()
        else:
            st.error(f"Query failed with return code {return_code}")
            st.code(stderr)
            return None


# Function to get available companies
def get_available_companies():
    """Get available companies from the database."""
    try:
        # Query the database for available companies
        companies = st.session_state.db_store.get_available_companies()
        return companies
    except Exception as e:
        st.error(f"Error getting available companies: {str(e)}")
        return ["NVDA"]  # Default to NVIDIA if there's an error


# Function to get available metrics
def get_available_metrics():
    """Get available metrics from the database."""
    try:
        # Query the database for available metrics
        metrics = st.session_state.db_store.get_available_metrics()
        return metrics
    except Exception as e:
        st.error(f"Error getting available metrics: {str(e)}")
        return ["Revenue", "NetIncome", "GrossProfit"]  # Default metrics if there's an error


# Function to get available years
def get_available_years():
    """Get available years from the database."""
    try:
        # Query the database for available years
        years = st.session_state.db_store.get_available_years()
        return years
    except Exception as e:
        st.error(f"Error getting available years: {str(e)}")
        return list(range(2020, 2025))  # Default to 2020-2024 if there's an error


# Main app
def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<div class="main-header">SEC Filing Analyzer Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze SEC filings with ease</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown('<div class="sub-header">Controls</div>', unsafe_allow_html=True)

    # Get available companies
    available_companies = get_available_companies()

    # Company selection
    selected_company = st.sidebar.selectbox(
        "Select Company",
        available_companies,
        index=available_companies.index("NVDA") if "NVDA" in available_companies else 0,
    )

    # Year selection for ETL
    available_years = get_available_years()
    selected_years = st.sidebar.multiselect("Select Years for ETL", available_years, default=[2023])

    # Test mode checkbox
    test_mode = st.sidebar.checkbox("Use Synthetic Data (Test Mode)", value=False)

    # ETL button
    if st.sidebar.button("Run ETL Process"):
        run_etl(selected_company, selected_years, test_mode)

    # Divider
    st.sidebar.markdown("---")

    # Query controls
    st.sidebar.markdown('<div class="sub-header">Query Controls</div>', unsafe_allow_html=True)

    # Get available metrics
    available_metrics = get_available_metrics()

    # Metric selection
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics",
        available_metrics,
        default=["Revenue", "NetIncome"]
        if "Revenue" in available_metrics and "NetIncome" in available_metrics
        else available_metrics[:2],
    )

    # Year range selection for query
    if available_years and all(isinstance(y, (int, float)) for y in available_years):
        min_year = min(available_years)
        max_year = max(available_years)
    else:
        min_year = 2020
        max_year = 2024

    start_year, end_year = st.sidebar.slider(
        "Select Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year)
    )

    # Query button
    if st.sidebar.button("Query Financial Data"):
        st.session_state.selected_company = selected_company
        st.session_state.metrics = selected_metrics

        # Query financial data
        with st.spinner("Querying financial data..."):
            df = query_financial_data(selected_company, selected_metrics, start_year, end_year)

            if not df.empty:
                st.session_state.df = df
                st.success("Query successful!")
            else:
                st.warning("No data found for the selected criteria.")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="sub-header">Financial Data Visualization</div>', unsafe_allow_html=True)

        if "df" in st.session_state and not st.session_state.df.empty:
            # Pivot the DataFrame for plotting
            df_pivot = st.session_state.df.pivot(index="period_end_date", columns="metric_name", values="value")

            # Plot the data
            fig = go.Figure()

            for metric in df_pivot.columns:
                fig.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot[metric], mode="lines+markers", name=metric))

            fig.update_layout(
                title=f"Financial Metrics for {st.session_state.selected_company}",
                xaxis_title="Period End Date",
                yaxis_title="Value (USD)",
                legend_title="Metrics",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show the data in a table
            st.markdown('<div class="sub-header">Financial Data Table</div>', unsafe_allow_html=True)

            # Format the DataFrame for display
            display_df = st.session_state.df.copy()
            display_df["period_end_date"] = display_df["period_end_date"].dt.strftime("%Y-%m-%d")
            display_df["value"] = display_df["value"].apply(lambda x: f"${x:.2f} B" if x >= 1 else f"${x * 1000:.2f} M")

            # Rename columns
            display_df = display_df.rename(
                columns={
                    "ticker": "Ticker",
                    "metric_name": "Metric",
                    "value": "Value",
                    "period_end_date": "Period End Date",
                    "source": "Source",
                }
            )

            st.dataframe(display_df, use_container_width=True)
        else:
            st.markdown(
                '<div class="info-box">Select a company and metrics, then click "Query Financial Data" to see the visualization.</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown('<div class="sub-header">Quick Revenue Lookup</div>', unsafe_allow_html=True)

        # Year selection for revenue query
        revenue_year = st.selectbox(
            "Select Year", available_years, index=available_years.index(2023) if 2023 in available_years else 0
        )

        # Revenue query button
        if st.button("Get Revenue"):
            with st.spinner(f"Getting revenue for {selected_company} in {revenue_year}..."):
                revenue = query_revenue(selected_company, revenue_year)

                if revenue:
                    st.markdown(f'<div class="success-box">{revenue}</div>', unsafe_allow_html=True)

        # ETL Status
        st.markdown('<div class="sub-header">ETL Status</div>', unsafe_allow_html=True)

        if st.session_state.etl_complete:
            st.markdown(
                f'<div class="success-box">ETL process for {st.session_state.selected_company} completed successfully!</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warning-box">ETL process not yet run. Click "Run ETL Process" to start.</div>',
                unsafe_allow_html=True,
            )

        # About the Demo
        st.markdown('<div class="sub-header">About the Demo</div>', unsafe_allow_html=True)

        st.markdown(
            """
        <div class="info-box">
        This demo showcases the core functionality of the SEC Filing Analyzer:

        1. **ETL Process**: Extract, transform, and load SEC filings
        2. **Financial Data Query**: Query financial data from the database
        3. **Visualization**: Visualize financial metrics over time

        The demo uses real SEC filings from the EDGAR database, with an option to use synthetic data for testing.
        </div>
        """,
            unsafe_allow_html=True,
        )


def run():
    """Run the Streamlit app."""
    main()


if __name__ == "__main__":
    main()
