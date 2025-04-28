"""
ETL Pipeline Page

This page provides a user interface for configuring and running the ETL pipeline.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Import ETL components
from sec_filing_analyzer.config import ConfigProvider, ETLConfig
from sec_filing_analyzer.data_retrieval import SECFilingsDownloader
from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline
from sec_filing_analyzer.storage import GraphStore, LlamaIndexVectorStore

# Set page config
st.set_page_config(
    page_title="ETL Pipeline - SEC Filing Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize configuration
ConfigProvider.initialize()
etl_config = ConfigProvider.get_config(ETLConfig)

# Title and description
st.title("ETL Pipeline")
st.markdown("""
Configure and run the ETL pipeline to extract, transform, and load SEC filings data.
The pipeline can process both semantic (text) and quantitative (XBRL) data.
""")

# Sidebar for configuration
st.sidebar.header("Pipeline Configuration")

# Company selection
st.sidebar.subheader("Company Selection")
ticker_input = st.sidebar.text_input("Ticker Symbol(s)", "AAPL, MSFT, GOOGL")
tickers = [ticker.strip() for ticker in ticker_input.split(",") if ticker.strip()]

# Filing type selection
st.sidebar.subheader("Filing Types")
filing_types = st.sidebar.multiselect(
    "Select Filing Types",
    ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"],
    default=["10-K", "10-Q"],
)

# Date range selection
st.sidebar.subheader("Date Range")
today = datetime.now()
one_year_ago = today - timedelta(days=365)

start_date = st.sidebar.date_input("Start Date", one_year_ago)

end_date = st.sidebar.date_input("End Date", today)

# Processing options
st.sidebar.subheader("Processing Options")
process_semantic = st.sidebar.checkbox("Process Semantic Data", value=True)
process_quantitative = st.sidebar.checkbox("Process Quantitative Data", value=True)
use_parallel = st.sidebar.checkbox("Use Parallel Processing", value=True)

if use_parallel:
    max_workers = st.sidebar.slider("Max Workers", min_value=1, max_value=16, value=4)
    batch_size = st.sidebar.slider("Batch Size", min_value=10, max_value=500, value=100)
    rate_limit = st.sidebar.slider("Rate Limit (seconds)", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    db_path = st.text_input("DuckDB Path", value="data/financial_data.duckdb")
    force_download = st.checkbox("Force Download", value=False)
    limit_per_company = st.number_input("Limit Per Company", min_value=1, max_value=100, value=10)

# Main content
tab1, tab2, tab3 = st.tabs(["Configuration", "Execution", "Results"])

with tab1:
    st.header("Pipeline Configuration")

    # Display configuration summary
    st.subheader("Configuration Summary")

    config_data = {
        "Parameter": [
            "Tickers",
            "Filing Types",
            "Date Range",
            "Process Semantic",
            "Process Quantitative",
            "Use Parallel",
            "Max Workers",
            "Batch Size",
            "Rate Limit",
            "DuckDB Path",
            "Force Download",
            "Limit Per Company",
        ],
        "Value": [
            ", ".join(tickers),
            ", ".join(filing_types),
            f"{start_date} to {end_date}",
            "Yes" if process_semantic else "No",
            "Yes" if process_quantitative else "No",
            "Yes" if use_parallel else "No",
            max_workers if use_parallel else "N/A",
            batch_size if use_parallel else "N/A",
            rate_limit if use_parallel else "N/A",
            db_path,
            "Yes" if force_download else "No",
            limit_per_company,
        ],
    }

    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True)

    # Save configuration
    if st.button("Save Configuration"):
        st.success("Configuration saved successfully!")

with tab2:
    st.header("Pipeline Execution")

    # Check if tickers are provided
    if not tickers:
        st.error("Please enter at least one ticker symbol.")
    else:
        # Initialize pipeline
        if st.button("Initialize Pipeline"):
            with st.spinner("Initializing pipeline..."):
                # Create pipeline instance
                try:
                    # Initialize components
                    st.session_state.downloader = SECFilingsDownloader()
                    st.session_state.graph_store = GraphStore()
                    st.session_state.vector_store = LlamaIndexVectorStore()

                    # Initialize pipeline
                    st.session_state.pipeline = SECFilingETLPipeline(
                        graph_store=st.session_state.graph_store,
                        vector_store=st.session_state.vector_store,
                        sec_downloader=st.session_state.downloader,
                        max_workers=max_workers if use_parallel else 1,
                        batch_size=batch_size,
                        rate_limit=rate_limit,
                        use_parallel=use_parallel,
                        process_semantic=process_semantic,
                        process_quantitative=process_quantitative,
                        db_path=db_path,
                    )

                    st.success("Pipeline initialized successfully!")
                    st.session_state.pipeline_initialized = True
                except Exception as e:
                    st.error(f"Error initializing pipeline: {str(e)}")
                    st.session_state.pipeline_initialized = False

        # Run pipeline
        if st.button("Run Pipeline") and st.session_state.get("pipeline_initialized", False):
            with st.spinner("Running pipeline..."):
                try:
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Process each company
                    results = {}
                    for i, ticker in enumerate(tickers):
                        status_text.text(f"Processing {ticker}...")

                        # Process company filings
                        result = st.session_state.pipeline.process_company_filings(
                            ticker=ticker,
                            filing_types=filing_types,
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d"),
                            limit=limit_per_company,
                            force_download=force_download,
                        )

                        results[ticker] = result

                        # Update progress
                        progress = (i + 1) / len(tickers)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {i + 1} of {len(tickers)} companies")

                    # Store results in session state
                    st.session_state.etl_results = results

                    # Complete
                    progress_bar.progress(1.0)
                    status_text.text("Pipeline execution completed!")
                    st.success("Pipeline execution completed successfully!")

                except Exception as e:
                    st.error(f"Error running pipeline: {str(e)}")

with tab3:
    st.header("Pipeline Results")

    if "etl_results" in st.session_state:
        # Display results
        st.subheader("Processing Results")

        # Create a summary table
        summary_data = []

        for ticker, result in st.session_state.etl_results.items():
            if "error" in result:
                status = "Error"
                details = result["error"]
                num_filings = 0
            else:
                status = "Success"
                details = result.get("status", "Completed")
                num_filings = len(result.get("results", []))

            summary_data.append(
                {
                    "Ticker": ticker,
                    "Status": status,
                    "Filings Processed": num_filings,
                    "Details": details,
                }
            )

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # Detailed results
        st.subheader("Detailed Results")

        selected_ticker = st.selectbox("Select Ticker", list(st.session_state.etl_results.keys()))

        if selected_ticker:
            result = st.session_state.etl_results[selected_ticker]

            if "error" in result:
                st.error(f"Error processing {selected_ticker}: {result['error']}")
            else:
                # Display filings processed
                filings = result.get("results", [])

                if filings:
                    filings_data = []

                    for filing in filings:
                        filings_data.append(
                            {
                                "Filing Type": filing.get("filing_type", "Unknown"),
                                "Filing Date": filing.get("filing_date", "Unknown"),
                                "Accession Number": filing.get("accession_number", "Unknown"),
                                "Status": filing.get("status", "Unknown"),
                                "Semantic Processing": "Yes" if filing.get("semantic_processed", False) else "No",
                                "Quantitative Processing": "Yes"
                                if filing.get("quantitative_processed", False)
                                else "No",
                            }
                        )

                    filings_df = pd.DataFrame(filings_data)
                    st.dataframe(filings_df, use_container_width=True)
                else:
                    st.info(f"No filings processed for {selected_ticker}")
    else:
        st.info("No pipeline results available. Run the pipeline first.")
