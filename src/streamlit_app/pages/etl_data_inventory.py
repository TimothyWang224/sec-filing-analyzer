"""
ETL Data Inventory

This page shows what companies and filings are currently stored in the system.
"""

import streamlit as st
import pandas as pd
import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import duckdb
import time

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Import utility functions
from src.streamlit_app.utils import launch_duckdb_ui, app_state

# Import the ETL service
from src.streamlit_app.services import get_etl_service

# Import the DuckDB manager
from src.sec_filing_analyzer.utils.duckdb_manager import duckdb_manager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('etl_data_inventory.log')
    ]
)
logger = logging.getLogger('etl_data_inventory')

# Log startup information
logger.info("ETL Data Inventory starting up")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")

# Initialize the ETL service
etl_service = get_etl_service()

# Set page config
st.set_page_config(
    page_title="ETL Data Inventory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for refresh tracking
if 'last_data_load' not in st.session_state:
    st.session_state.last_data_load = datetime.now()
if 'refresh_duration' not in st.session_state:
    st.session_state.refresh_duration = 0.0
if 'is_refreshing' not in st.session_state:
    st.session_state.is_refreshing = False
if 'refresh_start_time' not in st.session_state:
    st.session_state.refresh_start_time = 0
if 'refresh_id' not in st.session_state:
    st.session_state.refresh_id = 0

# Function to refresh data
def refresh_data():
    # Set the refreshing flag and start time
    st.session_state.is_refreshing = True
    st.session_state.refresh_start_time = time.time()

    # Generate a new refresh ID to invalidate the cache
    st.session_state.refresh_id = time.time()

    # Force a rerun to refresh the data
    st.rerun()

# Function to track data load time
def track_data_load():
    # Only update timing if this is a refresh operation
    if st.session_state.is_refreshing:
        # Calculate duration
        duration = time.time() - st.session_state.refresh_start_time
        st.session_state.refresh_duration = round(duration, 2)

        # Reset the refreshing flag
        st.session_state.is_refreshing = False

        # Update last data load timestamp
        st.session_state.last_data_load = datetime.now()
    elif 'first_load' not in st.session_state:
        # If this is the first load, update the timestamp
        st.session_state.last_data_load = datetime.now()
        st.session_state.first_load = True

# Title and description
st.title("ETL Data Inventory")
st.markdown("""
This page shows what companies and filings are currently stored in the system.
Use this to see what data is available and identify what new data to add.
""")

# Function to get data from DuckDB
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_duckdb_data(db_path="data/db_backup/improved_financial_data.duckdb", _refresh_id=None):
    """Get data from DuckDB database."""
    if not os.path.exists(db_path):
        logger.warning(f"DuckDB database not found at {db_path}")
        return None, None, None

    try:
        # Connect to DuckDB in read-only mode using the DuckDB manager
        conn = duckdb_manager.get_read_only_connection(db_path)
        logger.info(f"Connected to DuckDB database at {db_path} in read-only mode")

        # Get companies
        companies_df = conn.execute("SELECT * FROM companies").fetchdf()
        logger.info(f"Retrieved {len(companies_df)} companies from DuckDB")

        # Get filings with improved schema
        # First check if the table exists and what columns it has
        table_info = conn.execute("PRAGMA table_info(filings)").fetchdf()
        column_names = table_info['name'].tolist() if not table_info.empty else []

        # Build a query based on available columns
        if 'filing_id' in column_names:
            # Improved schema
            filings_query = """
            SELECT
                filing_id, company_id, accession_number, filing_type, filing_date,
                fiscal_year,
            """

            # Add fiscal_period if it exists
            if 'fiscal_period' in column_names:
                filings_query += "fiscal_period, "
            elif 'fiscal_quarter' in column_names:
                filings_query += "fiscal_quarter as fiscal_period, "
            else:
                filings_query += "NULL as fiscal_period, "

            filings_query += """
                fiscal_period_end_date, document_url,
                has_xbrl, created_at,
            """

            # Add updated_at if it exists
            if 'updated_at' in column_names:
                filings_query += "updated_at"
            else:
                filings_query += "created_at as updated_at"

            filings_query += """
            FROM filings
            """
        else:
            # Legacy schema
            filings_query = """
            SELECT
                id as filing_id,
                ticker as company_id,
                accession_number,
                filing_type,
                filing_date,
                fiscal_year,
                fiscal_quarter as fiscal_period,
                fiscal_period_end_date,
                document_url,
                has_xbrl,
                created_at,
                created_at as updated_at
            FROM filings
            """
        filings_df = conn.execute(filings_query).fetchdf()
        logger.info(f"Retrieved {len(filings_df)} filings from DuckDB")

        # Get filing counts by company, type, and fiscal period
        # Build a query based on available columns
        if 'company_id' in column_names and 'ticker' in conn.execute("PRAGMA table_info(companies)").fetchdf()['name'].tolist():
            # Improved schema with company_id foreign key
            filing_counts_query = """
            SELECT
                c.ticker,
                c.name as company_name,
                f.filing_type,
            """

            # Add fiscal_period if it exists
            if 'fiscal_period' in column_names:
                filing_counts_query += """
                f.fiscal_period,
                """
            elif 'fiscal_quarter' in column_names:
                filing_counts_query += """
                f.fiscal_quarter as fiscal_period,
                """
            else:
                filing_counts_query += """
                'Unknown' as fiscal_period,
                """

            filing_counts_query += """
                COUNT(*) as count,
                MIN(f.filing_date) as earliest_date,
                MAX(f.filing_date) as latest_date
            FROM
                filings f
            JOIN
                companies c ON f.company_id = c.company_id
            GROUP BY
                c.ticker, c.name, f.filing_type,
            """

            # Add fiscal_period to GROUP BY if it exists
            if 'fiscal_period' in column_names:
                filing_counts_query += "f.fiscal_period"
            elif 'fiscal_quarter' in column_names:
                filing_counts_query += "f.fiscal_quarter"
            else:
                filing_counts_query += "'Unknown'"

            filing_counts_query += """
            ORDER BY
                c.ticker, f.filing_type
            """
        else:
            # Legacy schema with ticker as foreign key
            filing_counts_query = """
            SELECT
                f.ticker,
                c.name as company_name,
                f.filing_type,
                COALESCE(f.fiscal_quarter, 'Unknown') as fiscal_period,
                COUNT(*) as count,
                MIN(f.filing_date) as earliest_date,
                MAX(f.filing_date) as latest_date
            FROM
                filings f
            JOIN
                companies c ON f.ticker = c.ticker
            GROUP BY
                f.ticker, c.name, f.filing_type, COALESCE(f.fiscal_quarter, 'Unknown')
            ORDER BY
                f.ticker, f.filing_type
            """
        filing_counts_df = conn.execute(filing_counts_query).fetchdf()
        logger.info(f"Retrieved filing counts by company, type, and processing status")

        # Note: We don't close the connection since it's managed by the DuckDB manager
        # and may be reused by other parts of the application

        return companies_df, filings_df, filing_counts_df
    except Exception as e:
        logger.error(f"Error retrieving data from DuckDB: {e}")
        return None, None, None

# Function to get data from vector store
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_vector_store_data(vector_store_path="data/vector_store", _refresh_id=None):
    """Get data from vector store."""
    if not os.path.exists(vector_store_path):
        logger.warning(f"Vector store not found at {vector_store_path}")
        return None

    try:
        # Get company directories
        company_dir = Path(vector_store_path) / "by_company"
        if not company_dir.exists():
            logger.warning(f"Company directory not found at {company_dir}")
            return None

        # Get companies with embeddings
        companies = [d.name for d in company_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(companies)} companies in vector store")

        # Get embedding counts by company
        embedding_counts = {}
        for company in companies:
            company_path = company_dir / company
            embedding_files = list(company_path.glob("*.npy"))
            embedding_counts[company] = len(embedding_files)

        logger.info(f"Retrieved embedding counts for {len(embedding_counts)} companies")

        # Check if index exists
        index_dir = Path(vector_store_path) / "index"
        index_exists = index_dir.exists() and (index_dir / "index").exists()

        return {
            "companies": companies,
            "embedding_counts": embedding_counts,
            "path": vector_store_path,
            "index_exists": index_exists
        }
    except Exception as e:
        logger.error(f"Error retrieving data from vector store: {e}")
        return None

# Function to get inventory summary
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_inventory_summary(_refresh_id=None):
    """Get inventory summary from ETL service."""
    try:
        # Get inventory summary from ETL service
        summary = etl_service.get_inventory_summary()
        logger.info(f"Retrieved inventory summary: {len(summary.get('company_counts', []))} companies, {summary.get('total_filings', 0)} filings")
        return summary
    except Exception as e:
        logger.error(f"Error retrieving inventory summary: {e}")
        return None

# Function to sync storage
def sync_storage():
    """Synchronize storage systems."""
    try:
        # Sync storage using ETL service
        results = etl_service.sync_storage()
        logger.info(f"Storage synchronization completed: {results}")
        return results
    except Exception as e:
        logger.error(f"Error synchronizing storage: {e}")
        return {"error": str(e)}

# Main content
# Sidebar filters
st.sidebar.header("Filters")

# Processing status filter
st.sidebar.subheader("Processing Status")
all_processing_statuses = ["downloaded", "processed", "embedded", "xbrl_processed", "error", "unknown"]
selected_processing_statuses = st.sidebar.multiselect(
    "Select Processing Status",
    options=all_processing_statuses,
    default=all_processing_statuses
)

# Date range filter
st.sidebar.subheader("Date Range")
today = datetime.now()
default_start_date = today - timedelta(days=365*2)  # 2 years ago
default_end_date = today

start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", default_end_date)

# Add actions section
st.sidebar.subheader("Actions")

# Check if index needs rebuilding
index_needs_rebuild = app_state.get("index_needs_rebuild", False)

# Show warning if index needs rebuilding
if index_needs_rebuild:
    st.sidebar.warning("⚠️ New data has been added. Please rebuild the vector index for complete search results.")

# Add a refresh button
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.session_state.refresh_id = str(time.time())
    st.rerun()

# Add a button to force rebuild the vector store index
if st.sidebar.button("🔄 Rebuild Vector Index", use_container_width=True):
    try:
        # Get the ETL service
        etl_service = get_etl_service()

        # Force rebuild the index
        if etl_service and etl_service.pipeline and etl_service.pipeline.vector_store:
            # Create a new vector store with force_rebuild=True
            from sec_filing_analyzer.storage.vector_store import LlamaIndexVectorStore
            from sec_filing_analyzer.config import ConfigProvider, StorageConfig

            # Get the storage config
            ConfigProvider.initialize()
            storage_config = ConfigProvider.get_config(StorageConfig)

            # Create a new vector store with force_rebuild=True
            vector_store = LlamaIndexVectorStore(
                store_path=storage_config.vector_store_path,
                force_rebuild=True,  # Force rebuild the index
                lazy_load=False  # Don't use lazy loading when explicitly rebuilding
            )

            # Replace the vector store in the pipeline
            etl_service.pipeline.vector_store = vector_store

            # Clear the rebuild flag
            app_state.set("index_needs_rebuild", False)

            st.sidebar.success("Vector index rebuilt successfully!")

            # Refresh the data
            st.session_state.refresh_id = str(time.time())
            st.rerun()
        else:
            st.sidebar.error("ETL service or pipeline not initialized.")
    except Exception as e:
        st.sidebar.error(f"Error rebuilding vector index: {e}")

# Get data with refresh ID to invalidate cache when refreshing
companies_df, filings_df, filing_counts_df = get_duckdb_data(_refresh_id=st.session_state.refresh_id)
vector_store_data = get_vector_store_data(_refresh_id=st.session_state.refresh_id)

# Track data load time
track_data_load()

# Check if vector store index exists and show status
if vector_store_data and 'index_exists' in vector_store_data:
    index_exists = vector_store_data['index_exists']
    if not index_exists and not index_needs_rebuild:
        st.sidebar.warning("⚠️ Vector index not found. Please rebuild the vector index for search functionality.")
    elif index_exists and not index_needs_rebuild:
        st.sidebar.success("✅ Vector index is up to date.")

# Display data
if companies_df is not None and filings_df is not None and filing_counts_df is not None:
    # Convert date strings to datetime objects for filtering
    if 'filing_date' in filings_df.columns:
        filings_df['filing_date'] = pd.to_datetime(filings_df['filing_date'])

    if 'earliest_date' in filing_counts_df.columns and 'latest_date' in filing_counts_df.columns:
        filing_counts_df['earliest_date'] = pd.to_datetime(filing_counts_df['earliest_date'])
        filing_counts_df['latest_date'] = pd.to_datetime(filing_counts_df['latest_date'])

    # Apply date filters
    if 'filing_date' in filings_df.columns:
        filtered_filings_df = filings_df[
            (filings_df['filing_date'] >= pd.Timestamp(start_date)) &
            (filings_df['filing_date'] <= pd.Timestamp(end_date))
        ]
    else:
        filtered_filings_df = filings_df

    # Apply fiscal period filter instead of processing status
    if 'fiscal_period' in filtered_filings_df.columns and selected_processing_statuses:
        # Map processing statuses to fiscal periods for compatibility
        fiscal_period_map = {
            "downloaded": "Q1",
            "processed": "Q2",
            "embedded": "Q3",
            "xbrl_processed": "Q4",
            "error": "FY",
            "unknown": None
        }
        selected_fiscal_periods = [fiscal_period_map.get(status) for status in selected_processing_statuses if fiscal_period_map.get(status) is not None]

        if selected_fiscal_periods:
            filtered_filings_df = filtered_filings_df[
                filtered_filings_df['fiscal_period'].isin(selected_fiscal_periods)
            ]

    # Filter filing counts
    if 'latest_date' in filing_counts_df.columns:
        filtered_counts_df = filing_counts_df[
            (filing_counts_df['latest_date'] >= pd.Timestamp(start_date)) &
            (filing_counts_df['earliest_date'] <= pd.Timestamp(end_date))
        ]
    else:
        filtered_counts_df = filing_counts_df

    # Apply fiscal period filter to filing counts instead of processing status
    if 'fiscal_period' in filtered_counts_df.columns and selected_processing_statuses:
        # Map processing statuses to fiscal periods for compatibility
        fiscal_period_map = {
            "downloaded": "Q1",
            "processed": "Q2",
            "embedded": "Q3",
            "xbrl_processed": "Q4",
            "error": "FY",
            "unknown": None
        }
        selected_fiscal_periods = [fiscal_period_map.get(status) for status in selected_processing_statuses if fiscal_period_map.get(status) is not None]

        if selected_fiscal_periods:
            filtered_counts_df = filtered_counts_df[
                filtered_counts_df['fiscal_period'].isin(selected_fiscal_periods)
            ]

    # Display summary
    col1, col2, col3, col4, col5 = st.columns([5, 3, 3, 3, 3])

    # Check if vector store index exists
    index_exists = False
    if vector_store_data and 'index_exists' in vector_store_data:
        index_exists = vector_store_data['index_exists']

    with col1:
        st.header("Data Summary")

    with col2:
        # Display last data load timestamp
        st.markdown(f"**Last data load:** {st.session_state.last_data_load.strftime('%Y-%m-%d %H:%M:%S')}")

    with col3:
        # Display refresh duration if available
        if st.session_state.refresh_duration > 0:
            st.markdown(f"**Refresh time:** {st.session_state.refresh_duration} seconds")

    with col4:
        # Add refresh button for data summary
        if st.button("🔄 Refresh Summary", key="refresh_summary", help="Refresh the data summary to see the latest changes"):
            refresh_data()

    with col5:
        # Add DuckDB UI button
        if st.button("🔍 Open DuckDB UI", key="open_duckdb_ui", help="Open the DuckDB UI in a new browser tab"):
            # Launch DuckDB UI using the utility function
            launch_duckdb_ui()

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Companies", len(companies_df))

    with summary_col2:
        st.metric("Total Filings", len(filings_df))

    with summary_col3:
        st.metric("Filtered Filings", len(filtered_filings_df))

    # Display filing counts by company and type
    st.header("Filings by Company and Type")

    if not filtered_counts_df.empty:
        # Print column names for debugging
        st.write("Available columns in filtered_counts_df:", list(filtered_counts_df.columns))

        # Check if 'company_name' column exists
        if 'company_name' in filtered_counts_df.columns:
            # Use company_name directly
            companies = filtered_counts_df[['ticker', 'company_name']].drop_duplicates()
        elif 'name' in filtered_counts_df.columns:
            # Use name and rename to company_name
            companies = filtered_counts_df[['ticker', 'name']].drop_duplicates()
            # Rename 'name' to 'company_name' for consistency
            companies = companies.rename(columns={'name': 'company_name'})
        else:
            # Create a fallback with just ticker
            st.warning("Company name column not found in the data. Using ticker only.")
            companies = filtered_counts_df[['ticker']].drop_duplicates()
            # Add a placeholder company_name column
            companies['company_name'] = companies['ticker'] + " (Unknown)"

        for _, company_row in companies.iterrows():
            ticker = company_row['ticker']
            company_name = company_row['company_name']

            # Create an expander for each company
            with st.expander(f"{ticker} - {company_name}"):
                # Get filing types for this company
                company_filings = filtered_counts_df[filtered_counts_df['ticker'] == ticker]

                # Display as a table
                st.dataframe(
                    company_filings[['filing_type', 'count', 'earliest_date', 'latest_date']],
                    use_container_width=True
                )

                # Add a button to view details
                if st.button(f"View Details for {ticker}", key=f"details_{ticker}"):
                    st.subheader(f"Detailed Filings for {ticker}")
                    company_detailed_filings = filtered_filings_df[filtered_filings_df['ticker'] == ticker]
                    st.dataframe(company_detailed_filings, use_container_width=True)
    else:
        st.warning("No filings found for the selected date range.")

    # Display raw data in tabs
    st.header("Raw Data")
    tab1, tab2, tab3 = st.tabs(["Companies", "Filings", "Filing Counts"])

    with tab1:
        st.subheader("Companies")
        st.dataframe(companies_df, use_container_width=True)

    with tab2:
        st.subheader("Filings")
        st.dataframe(filtered_filings_df, use_container_width=True)

    with tab3:
        st.subheader("Filing Counts")
        st.dataframe(filtered_counts_df, use_container_width=True)
else:
    st.warning("No data available. Please run the ETL pipeline to extract data.")

# Display filings by company and time period
col1, col2, col3 = st.columns([6, 3, 3])
with col1:
    st.header("Filings Inventory")
with col2:
    # Display last data load timestamp and refresh info
    st.markdown(f"**Last data load:** {st.session_state.last_data_load.strftime('%Y-%m-%d %H:%M:%S')}")
    if st.session_state.refresh_duration > 0:
        st.markdown(f"**Refresh time:** {st.session_state.refresh_duration} seconds")
with col3:
    # Add refresh button
    if st.button("🔄 Refresh Data", key="refresh_inventory", help="Refresh the data to see the latest changes"):
        refresh_data()

# Create tabs for different views
inventory_tabs = st.tabs(["By Company", "By Filing Type", "By Date"])

# Function to get file path for a filing
def get_filing_path(ticker, filing_type, filing_date, accession_number=None, local_file_path=None):
    # If local_file_path is provided, use it
    if local_file_path and os.path.exists(local_file_path):
        return local_file_path

    # Base path for SEC filings
    base_path = Path("data/filings")

    # Format date for folder structure (YYYY-MM-DD to YYYYMMDD)
    if isinstance(filing_date, str):
        formatted_date = filing_date.replace("-", "")
    else:
        # Handle datetime objects
        formatted_date = filing_date.strftime("%Y%m%d")

    # If accession number is provided, use it
    if accession_number:
        # Try different file extensions
        for ext in ["html", "txt", "xml", "json"]:
            # Try different directory structures
            paths_to_try = [
                # Raw filings
                base_path / "raw" / ticker / f"{accession_number}.{ext}",
                base_path / "raw" / ticker / str(formatted_date)[:4] / f"{accession_number}.{ext}",
                # HTML filings
                base_path / "html" / ticker / f"{accession_number}.{ext}",
                base_path / "html" / ticker / str(formatted_date)[:4] / f"{accession_number}.{ext}",
                # Processed filings
                base_path / "processed" / ticker / f"{accession_number}_processed.json",
                base_path / "processed" / ticker / str(formatted_date)[:4] / f"{accession_number}_processed.json",
                # XBRL filings
                base_path / "xbrl" / ticker / f"{accession_number}.{ext}",
                base_path / "xbrl" / ticker / str(formatted_date)[:4] / f"{accession_number}.{ext}",
            ]

            for path in paths_to_try:
                if os.path.exists(path):
                    return path
    else:
        # Otherwise create a generic name
        file_name = f"{ticker}_{filing_type}_{formatted_date}.html"

        # Construct the full path
        full_path = base_path / "html" / ticker / file_name

        # Check if file exists
        if os.path.exists(full_path):
            return full_path
        else:
            # Try alternative locations
            alt_paths = [
                base_path / "raw" / ticker / file_name,
                base_path / "html" / ticker / file_name,
                base_path / "processed" / ticker / file_name.replace(".html", "_processed.json"),
                base_path / f"{ticker}_{filing_type}_{formatted_date}.html"
            ]

            for path in alt_paths:
                if os.path.exists(path):
                    return path

    # If no file found, return None
    return None

# Function to check if a file exists
def check_file_exists(file_path):
    """Check if a file exists."""
    if not file_path:
        return False

    return os.path.exists(file_path)

# By Company tab
with inventory_tabs[0]:
    if companies_df is not None and filings_df is not None:
        # Group filings by company
        company_filings = {}

        # Process filings to group by company
        for _, filing in filings_df.iterrows():
            # Get ticker from either 'ticker' or 'company_id' field
            ticker = None
            if 'ticker' in filing:
                ticker = filing['ticker']
            elif 'company_id' in filing:
                ticker = filing['company_id']

            if ticker:
                if ticker not in company_filings:
                    company_filings[ticker] = []
                company_filings[ticker].append(filing)

        # Create expanders for each company
        for ticker, filings in company_filings.items():
            # Get company name
            company_name = "Unknown"
            company_row = companies_df[companies_df['ticker'] == ticker]
            if not company_row.empty:
                # Check which column exists
                if 'name' in company_row.columns:
                    company_name = company_row.iloc[0]['name']
                elif 'company_name' in company_row.columns:
                    company_name = company_row.iloc[0]['company_name']
                else:
                    company_name = ticker + " (Unknown)"

            # Create expander
            with st.expander(f"{ticker} - {company_name} ({len(filings)} filings)"):
                # Create a dataframe for this company's filings
                company_filings_data = []

                for filing in filings:
                    filing_date = filing['filing_date']
                    if isinstance(filing_date, pd.Timestamp):
                        filing_date = filing_date.date()

                    # Get filing details
                    accession_number = filing.get('accession_number', None)
                    document_url = filing.get('document_url', None)
                    fiscal_period = filing.get('fiscal_period', "unknown")

                    # Map fiscal period to a status for display
                    status_map = {
                        "Q1": "downloaded",
                        "Q2": "processed",
                        "Q3": "embedded",
                        "Q4": "xbrl_processed",
                        "FY": "completed",
                        None: "unknown"
                    }
                    processing_status = status_map.get(fiscal_period, "unknown")

                    # Check if file exists on disk
                    file_path = get_filing_path(ticker, filing['filing_type'], filing_date, accession_number, None)
                    file_exists = file_path is not None

                    # Create status icon based on processing status
                    status_icons = {
                        "downloaded": "📥",
                        "processed": "⚙️",
                        "embedded": "🔍",
                        "xbrl_processed": "📊",
                        "error": "❌",
                        "unknown": "❓"
                    }
                    status_icon = status_icons.get(processing_status, "❓")

                    company_filings_data.append({
                        "Filing Type": filing['filing_type'],
                        "Filing Date": filing_date,
                        "Accession Number": accession_number if accession_number else "N/A",
                        "Status": f"{status_icon} {processing_status}",
                        "On Disk": "✅" if file_exists else "❌",
                        "Path": str(file_path) if file_path else "Not found"
                    })

                # Convert to dataframe and display
                if company_filings_data:
                    df = pd.DataFrame(company_filings_data)
                    st.dataframe(df, use_container_width=True)

                    # Add a button to view files
                    if st.button(f"Open Files Directory for {ticker}", key=f"open_dir_{ticker}"):
                        # This would normally open the file explorer
                        st.info(f"In a production environment, this would open the file explorer to the directory containing {ticker}'s filings.")
                else:
                    st.info(f"No filings found for {ticker}.")
    else:
        st.warning("No company or filing data available. Please run the ETL pipeline to extract data.")

# By Filing Type tab
with inventory_tabs[1]:
    if filings_df is not None:
        # Group filings by type
        filing_types = filings_df['filing_type'].unique()

        # Create expanders for each filing type
        for filing_type in filing_types:
            # Get filings of this type
            type_filings = filings_df[filings_df['filing_type'] == filing_type]

            # Create expander
            with st.expander(f"{filing_type} ({len(type_filings)} filings)"):
                # Create a dataframe for this filing type
                type_filings_data = []

                for _, filing in type_filings.iterrows():
                    # Get ticker from either 'ticker' or 'company_id' field
                    ticker = None
                    if 'ticker' in filing:
                        ticker = filing['ticker']
                    elif 'company_id' in filing:
                        ticker = filing['company_id']
                    else:
                        # Skip this filing if we can't identify the company
                        continue

                    # Get filing date
                    if 'filing_date' in filing:
                        filing_date = filing['filing_date']
                        if isinstance(filing_date, pd.Timestamp):
                            filing_date = filing_date.date()
                    else:
                        filing_date = None

                    # Get company name
                    company_name = "Unknown"
                    company_row = companies_df[companies_df['ticker'] == ticker]
                    if not company_row.empty:
                        # Check which column exists
                        if 'name' in company_row.columns:
                            company_name = company_row.iloc[0]['name']
                        elif 'company_name' in company_row.columns:
                            company_name = company_row.iloc[0]['company_name']
                        else:
                            company_name = ticker + " (Unknown)"

                    # Get filing details
                    accession_number = filing.get('accession_number', None)
                    document_url = filing.get('document_url', None)
                    fiscal_period = filing.get('fiscal_period', "unknown")

                    # Map fiscal period to a status for display
                    status_map = {
                        "Q1": "downloaded",
                        "Q2": "processed",
                        "Q3": "embedded",
                        "Q4": "xbrl_processed",
                        "FY": "completed",
                        None: "unknown"
                    }
                    processing_status = status_map.get(fiscal_period, "unknown")

                    # Check if file exists on disk
                    file_path = get_filing_path(ticker, filing_type, filing_date, accession_number, None)
                    file_exists = file_path is not None

                    # Create status icon based on processing status
                    status_icons = {
                        "downloaded": "📥",
                        "processed": "⚙️",
                        "embedded": "🔍",
                        "xbrl_processed": "📊",
                        "error": "❌",
                        "unknown": "❓"
                    }
                    status_icon = status_icons.get(processing_status, "❓")

                    type_filings_data.append({
                        "Ticker": ticker,
                        "Company": company_name,
                        "Filing Date": filing_date,
                        "Accession Number": accession_number if accession_number else "N/A",
                        "Status": f"{status_icon} {processing_status}",
                        "On Disk": "✅" if file_exists else "❌",
                        "Path": str(file_path) if file_path else "Not found"
                    })

                # Convert to dataframe and display
                if type_filings_data:
                    df = pd.DataFrame(type_filings_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info(f"No {filing_type} filings found.")
    else:
        st.warning("No filing data available. Please run the ETL pipeline to extract data.")

# By Date tab
with inventory_tabs[2]:
    if filings_df is not None:
        # Convert filing_date to datetime if it's not already
        if 'filing_date' in filings_df.columns and not pd.api.types.is_datetime64_any_dtype(filings_df['filing_date']):
            filings_df['filing_date'] = pd.to_datetime(filings_df['filing_date'])

        # Group filings by year and quarter
        filings_df['year'] = filings_df['filing_date'].dt.year
        filings_df['quarter'] = (filings_df['filing_date'].dt.month - 1) // 3 + 1
        filings_df['year_quarter'] = filings_df['year'].astype(str) + "-Q" + filings_df['quarter'].astype(str)

        # Get unique year-quarters
        year_quarters = sorted(filings_df['year_quarter'].unique(), reverse=True)

        # Create expanders for each year-quarter
        for yq in year_quarters:
            # Get filings from this year-quarter
            yq_filings = filings_df[filings_df['year_quarter'] == yq]

            # Create expander
            with st.expander(f"{yq} ({len(yq_filings)} filings)"):
                # Create a dataframe for this year-quarter
                yq_filings_data = []

                for _, filing in yq_filings.iterrows():
                    # Get ticker from either 'ticker' or 'company_id' field
                    ticker = None
                    if 'ticker' in filing:
                        ticker = filing['ticker']
                    elif 'company_id' in filing:
                        ticker = filing['company_id']
                    else:
                        # Skip this filing if we can't identify the company
                        continue

                    # Get filing type
                    filing_type = filing.get('filing_type', 'Unknown')

                    # Get filing date
                    if 'filing_date' in filing:
                        filing_date = filing['filing_date']
                        if isinstance(filing_date, pd.Timestamp):
                            filing_date = filing_date.date()
                    else:
                        filing_date = None

                    # Get company name
                    company_name = "Unknown"
                    company_row = companies_df[companies_df['ticker'] == ticker]
                    if not company_row.empty:
                        # Check which column exists
                        if 'name' in company_row.columns:
                            company_name = company_row.iloc[0]['name']
                        elif 'company_name' in company_row.columns:
                            company_name = company_row.iloc[0]['company_name']
                        else:
                            company_name = ticker + " (Unknown)"

                    # Get filing details
                    accession_number = filing.get('accession_number', None)
                    local_file_path = filing.get('local_file_path', None)
                    processing_status = filing.get('processing_status', "unknown")

                    # Check if file exists on disk
                    file_path = get_filing_path(ticker, filing_type, filing_date, accession_number, local_file_path)
                    file_exists = file_path is not None

                    # Create status icon based on processing status
                    status_icons = {
                        "downloaded": "📥",
                        "processed": "⚙️",
                        "embedded": "🔍",
                        "xbrl_processed": "📊",
                        "error": "❌",
                        "unknown": "❓"
                    }
                    status_icon = status_icons.get(processing_status, "❓")

                    yq_filings_data.append({
                        "Ticker": ticker,
                        "Company": company_name,
                        "Filing Type": filing_type,
                        "Filing Date": filing_date,
                        "Accession Number": accession_number if accession_number else "N/A",
                        "Status": f"{status_icon} {processing_status}",
                        "On Disk": "✅" if file_exists else "❌",
                        "Path": str(file_path) if file_path else "Not found"
                    })

                # Convert to dataframe and display
                if yq_filings_data:
                    df = pd.DataFrame(yq_filings_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info(f"No filings found for {yq}.")
    else:
        st.warning("No filing data available. Please run the ETL pipeline to extract data.")

# ETL Staging Section
st.header("Stage ETL Retrieval")
st.markdown("""
Use this section to stage and run ETL retrieval for missing data.
Select the companies, filing types, and date range you want to retrieve.
""")

# Create columns for the selection controls
col1, col2 = st.columns(2)

with col1:
    # Company selection
    st.subheader("Select Companies")

    # Always use a text input field for company selection to allow for companies not in the database
    default_tickers = "AAPL, MSFT, GOOGL"

    # If companies exist in the database, show them as a reference
    if companies_df is not None and not companies_df.empty:
        # Display available companies as a reference
        with st.expander("Available Companies in Database"):
            # Check which column exists
            if 'name' in companies_df.columns:
                available_companies = companies_df[['ticker', 'name']].drop_duplicates()
                # Create a formatted list for display
                company_list = [f"{row['ticker']} - {row['name']}" for _, row in available_companies.iterrows()]
            elif 'company_name' in companies_df.columns:
                available_companies = companies_df[['ticker', 'company_name']].drop_duplicates()
                # Create a formatted list for display
                company_list = [f"{row['ticker']} - {row['company_name']}" for _, row in available_companies.iterrows()]
            else:
                # Just use tickers if no name column is available
                available_companies = companies_df[['ticker']].drop_duplicates()
                company_list = [f"{row['ticker']}" for _, row in available_companies.iterrows()]
            st.write("\n".join(company_list))

            # Add an "Use All Companies" button
            if st.button("Use All Companies"):
                default_tickers = ", ".join(list(companies_df['ticker'].unique()))
    else:
        st.info("No companies in database yet. Enter ticker symbols to retrieve data.")

    # Allow manual entry of tickers
    ticker_input = st.text_input("Enter ticker symbols (comma-separated)", default_tickers)
    selected_tickers = [ticker.strip() for ticker in ticker_input.split(",")]

with col2:
    # Filing type selection
    st.subheader("Select Filing Types")

    # Common filing types
    filing_types = ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"]

    # Add any additional filing types from the database
    if filings_df is not None and 'filing_type' in filings_df.columns:
        additional_types = list(filings_df['filing_type'].unique())
        filing_types = list(set(filing_types + additional_types))

    # Create checkboxes for filing types
    selected_filing_types = []

    # Create a "Select All" checkbox
    select_all = st.checkbox("Select All Filing Types", value=True)

    if select_all:
        selected_filing_types = filing_types
    else:
        # Create columns for the checkboxes to save space
        checkbox_cols = st.columns(3)
        for i, filing_type in enumerate(filing_types):
            with checkbox_cols[i % 3]:
                if st.checkbox(filing_type, value=False):
                    selected_filing_types.append(filing_type)

# Date range for ETL (dedicated controls)
st.subheader("Date Range for Retrieval")

# Initialize session state for date range if not already set
if 'retrieval_start_date' not in st.session_state:
    st.session_state.retrieval_start_date = start_date
if 'retrieval_end_date' not in st.session_state:
    st.session_state.retrieval_end_date = end_date
if 'selected_period_name' not in st.session_state:
    st.session_state.selected_period_name = "Custom Range"

# Create tabs for different date selection methods
date_tabs = st.tabs(["Quick Select", "Quarterly", "Custom Range"])

# Current date for calculations
today = datetime.now().date()

# Function to update date range
def update_date_range(start_date, end_date, period_name):
    st.session_state.retrieval_start_date = start_date
    st.session_state.retrieval_end_date = end_date
    st.session_state.selected_period_name = period_name
    return

# Quick Select tab
with date_tabs[0]:
    # Create buttons for common time periods
    st.write("Select a predefined time period:")

    # Create 3 columns for the buttons
    quick_col1, quick_col2, quick_col3 = st.columns(3)

    # Button for Last Year
    with quick_col1:
        if st.button("Last Year", use_container_width=True, key="btn_last_year"):
            last_year_start = datetime(today.year - 1, 1, 1).date()
            last_year_end = datetime(today.year - 1, 12, 31).date()
            update_date_range(last_year_start, last_year_end, "Last Year")

    # Button for Last 3 Years
    with quick_col2:
        if st.button("Last 3 Years", use_container_width=True, key="btn_last_3_years"):
            three_years_start = datetime(today.year - 3, 1, 1).date()
            update_date_range(three_years_start, today, "Last 3 Years")

    # Button for Last 5 Years
    with quick_col3:
        if st.button("Last 5 Years", use_container_width=True, key="btn_last_5_years"):
            five_years_start = datetime(today.year - 5, 1, 1).date()
            update_date_range(five_years_start, today, "Last 5 Years")

    # Second row of buttons
    quick_col4, quick_col5, quick_col6 = st.columns(3)

    # Button for Current Year
    with quick_col4:
        if st.button("Current Year", use_container_width=True, key="btn_current_year"):
            current_year_start = datetime(today.year, 1, 1).date()
            update_date_range(current_year_start, today, "Current Year")

    # Button for Current Quarter
    with quick_col5:
        if st.button("Current Quarter", use_container_width=True, key="btn_current_quarter"):
            current_quarter = (today.month - 1) // 3 + 1
            quarter_start_month = (current_quarter - 1) * 3 + 1
            quarter_start = datetime(today.year, quarter_start_month, 1).date()
            if current_quarter < 4:
                next_quarter_start = datetime(today.year, quarter_start_month + 3, 1).date()
            else:
                next_quarter_start = datetime(today.year + 1, 1, 1).date()
            quarter_end = (next_quarter_start - timedelta(days=1))
            update_date_range(quarter_start, quarter_end, f"Q{current_quarter} {today.year}")

    # Button for All Time
    with quick_col6:
        if st.button("All Time", use_container_width=True, key="btn_all_time"):
            all_time_start = datetime(1990, 1, 1).date()  # SEC EDGAR started in early 1990s
            update_date_range(all_time_start, today, "All Time")

# Quarterly tab
with date_tabs[1]:
    st.write("Select quarters and years:")

    # Year selection
    years = list(range(today.year, today.year - 10, -1))  # Last 10 years
    selected_years = st.multiselect("Years", years, default=[today.year, today.year - 1], key="quarterly_years")

    # Quarter selection
    quarters = ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"]
    selected_quarters = st.multiselect("Quarters", quarters, default=quarters, key="quarterly_quarters")

    # Convert selections to actual dates
    if selected_years and selected_quarters:
        # Find earliest and latest dates based on selections
        quarter_dates = []
        for year in selected_years:
            for quarter in selected_quarters:
                q_num = int(quarter[1])
                start_month = (q_num - 1) * 3 + 1
                if q_num < 4:
                    end_month = q_num * 3
                    end_day = 30 if end_month in [4, 6, 9] else 31
                    end_date = datetime(year, end_month, end_day).date()
                else:
                    end_date = datetime(year, 12, 31).date()
                start_date = datetime(year, start_month, 1).date()
                quarter_dates.append((start_date, end_date))

        if quarter_dates:
            quarterly_start = min([date[0] for date in quarter_dates])
            quarterly_end = max([date[1] for date in quarter_dates])

            # Display the selected date range
            st.write(f"Selected date range: {quarterly_start} to {quarterly_end}")

            # Show the quarters in a more readable format
            quarters_text = ", ".join([f"{q.split()[0]} {y}" for y in sorted(selected_years) for q in selected_quarters])
            st.write(f"Selected quarters: {quarters_text}")

            # Button to apply quarterly selection
            if st.button("Apply Quarterly Selection", key="apply_quarterly"):
                update_date_range(quarterly_start, quarterly_end, f"Quarterly: {quarters_text}")
    else:
        st.warning("Please select at least one year and one quarter.")

# Custom Range tab
with date_tabs[2]:
    st.write("Specify exact date range:")

    # Create two columns for start and end date
    custom_col1, custom_col2 = st.columns(2)

    # Start date picker
    with custom_col1:
        custom_start_date = st.date_input("Start Date", value=st.session_state.retrieval_start_date, key="custom_start")

    # End date picker
    with custom_col2:
        custom_end_date = st.date_input("End Date", value=st.session_state.retrieval_end_date, key="custom_end")

    # Update retrieval dates if custom dates are changed
    if custom_start_date and custom_end_date:
        if custom_start_date <= custom_end_date:
            # Button to apply custom date range
            if st.button("Apply Custom Range", key="apply_custom"):
                update_date_range(custom_start_date, custom_end_date, "Custom Range")
        else:
            st.error("End date must be after start date.")

# Display the final selected date range with the period name
st.info(f"Selected Retrieval Period: **{st.session_state.selected_period_name}** ({st.session_state.retrieval_start_date} to {st.session_state.retrieval_end_date})")

# Use these variables for the rest of the code
retrieval_start_date = st.session_state.retrieval_start_date
retrieval_end_date = st.session_state.retrieval_end_date

# Processing options
st.subheader("Processing Options")

# Add option to force reprocessing
force_reprocessing = st.checkbox(
    "Force Reprocessing",
    value=False,
    help="If checked, existing filings will be reprocessed even if they already exist in the database."
)

# Preview and Run buttons
st.subheader("Actions")
col1, col2 = st.columns(2)

with col1:
    if st.button("Preview Selection", use_container_width=True):
        st.subheader("Retrieval Preview")

        # Display selected companies
        st.write("**Selected Companies:**")
        if len(selected_tickers) > 10:
            st.write(f"{len(selected_tickers)} companies selected")
            st.write(", ".join(selected_tickers[:5]) + "... and " + str(len(selected_tickers) - 5) + " more")
        else:
            st.write(", ".join(selected_tickers))

        # Display selected filing types
        st.write("**Selected Filing Types:**")
        st.write(", ".join(selected_filing_types))

        # Display date range
        st.write(f"**Date Range:** {retrieval_start_date} to {retrieval_end_date}")

        # Calculate estimated filings
        # Calculate estimated filings using the ETL service
        estimated_filings = etl_service.estimate_filings_count(
            tickers=selected_tickers,
            filing_types=selected_filing_types,
            start_date=retrieval_start_date.strftime("%Y-%m-%d"),
            end_date=retrieval_end_date.strftime("%Y-%m-%d")
        )

        # Display estimates
        st.write(f"**Estimated Filings:** ~{estimated_filings}")
        st.write(f"**Estimated Storage:** ~{estimated_filings * 2} MB")
        st.write(f"**Estimated Time:** ~{estimated_filings // 10 + 1} minutes")

with col2:
    if st.button("Run ETL Pipeline", type="primary", use_container_width=True):
        # Calculate estimated filings using the ETL service
        estimated_filings = etl_service.estimate_filings_count(
            tickers=selected_tickers,
            filing_types=selected_filing_types,
            start_date=retrieval_start_date.strftime("%Y-%m-%d"),
            end_date=retrieval_end_date.strftime("%Y-%m-%d")
        )

        # Create a job in the ETL service
        job_id = etl_service.create_job(
            tickers=selected_tickers,
            filing_types=selected_filing_types,
            start_date=retrieval_start_date.strftime("%Y-%m-%d"),
            end_date=retrieval_end_date.strftime("%Y-%m-%d"),
            estimated_filings=estimated_filings,
            force_reprocessing=force_reprocessing
        )

        # Create a container for the ETL job details
        etl_container = st.container()

        with etl_container:
            # Show job details
            st.success("ETL Pipeline initiated!")
            st.write("**Job Details:**")
            st.write(f"- **Job ID:** {job_id}")
            st.write(f"- **Companies:** {', '.join(selected_tickers[:5])}{'...' if len(selected_tickers) > 5 else ''}")
            st.write(f"- **Filing Types:** {', '.join(selected_filing_types)}")
            st.write(f"- **Date Range:** {retrieval_start_date} to {retrieval_end_date}")
            st.write(f"- **Force Reprocessing:** {'Yes' if force_reprocessing else 'No'}")

            # Create a progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create a terminal-like display
            st.subheader("ETL Terminal Output")
            terminal_placeholder = st.empty()

            # Initialize terminal output
            terminal_output = ["[INFO] Starting ETL pipeline..."]

            # Start the job without a callback
            etl_service.start_job(job_id)

            # Poll for job status and update UI
            job_completed = False
            while not job_completed:
                # Get the job
                job = etl_service.get_job(job_id)

                if not job:
                    st.error(f"Job {job_id} not found")
                    break

                # Update progress bar and status text
                progress_bar.progress(job.progress)
                status_text.write(f"**Status:** {job.status} - {job.current_stage}")

                # Update terminal output with job logs
                logs = etl_service.get_job_logs(job_id)
                if logs:
                    # Format logs with timestamps
                    formatted_logs = []
                    for log in logs[-20:]:  # Show only the last 20 logs
                        formatted_logs.append(f"[INFO] {log}")

                    # Update terminal display
                    terminal_placeholder.code('\n'.join(formatted_logs), language="bash")

                # Check if job is completed or failed
                if job.status in ["Completed", "Failed"]:
                    job_completed = True

                    if job.status == "Completed":
                        # Show completion message
                        st.success("ETL Pipeline completed successfully!")
                        st.info("Refresh the page to see the updated inventory.")

                        # Check for any errors in the results
                        errors_found = False
                        error_details = []

                        for ticker, result in job.results.items():
                            if isinstance(result, dict) and "error" in result:
                                errors_found = True
                                error_details.append(f"**{ticker}**: {result['error']}")

                        # Display warnings if there were any errors
                        if errors_found:
                            with st.expander("⚠️ Some companies had errors", expanded=True):
                                st.warning("The ETL pipeline completed, but some companies had errors:")
                                for error in error_details:
                                    st.markdown(error)
                                st.info("The successful filings have been processed and are available in the inventory.")
                    else:
                        # Show error message
                        st.error(f"ETL Pipeline failed: {job.results.get('error', 'Unknown error')}")

                # Sleep for a short time before polling again
                time.sleep(1)

            # Show job details
            st.subheader("Job Details")
            st.json(job.to_dict())

# Data Management Section
st.header("Data Management")
st.markdown("""
Use this section to manage your data storage systems. You can synchronize data between different storage systems,
view inventory summary, and verify file existence.
""")

# Create tabs for different data management tasks
data_mgmt_tabs = st.tabs(["Storage Sync", "Inventory Summary", "Recent ETL Jobs"])

# Storage Sync tab
with data_mgmt_tabs[0]:
    st.subheader("Storage Synchronization")
    st.markdown("""
    Synchronize data between different storage systems (DuckDB, vector store, file system).
    This will ensure that all storage systems have consistent data.
    """)

    # Add a button to sync storage
    if st.button("Synchronize Storage", type="primary"):
        # Show a spinner while syncing
        with st.spinner("Synchronizing storage systems..."):
            # Sync storage
            results = sync_storage()

            # Display results
            if "error" in results:
                st.error(f"Error synchronizing storage: {results['error']}")
            elif "warning" in results:
                st.success("Storage synchronization partially completed!")
                st.warning(results["warning"])
                if "message" in results:
                    st.info(results["message"])
            else:
                st.success("Storage synchronization completed successfully!")
                if "message" in results:
                    st.info(results["message"])

                # Display summary
                st.subheader("Synchronization Results")

                # Vector store results
                st.write("**Vector Store:**")
                st.write(f"- Found: {results['vector_store']['found']}")
                st.write(f"- Added: {results['vector_store']['added']}")
                st.write(f"- Updated: {results['vector_store']['updated']}")
                st.write(f"- Errors: {results['vector_store']['errors']}")

                # File system results
                st.write("**File System:**")
                st.write(f"- Found: {results['file_system']['found']}")
                st.write(f"- Added: {results['file_system']['added']}")
                st.write(f"- Updated: {results['file_system']['updated']}")
                st.write(f"- Errors: {results['file_system']['errors']}")

                # Path update results
                st.write("**Path Updates:**")
                st.write(f"- Updated: {results['path_update']['updated']}")
                st.write(f"- Not Found: {results['path_update']['not_found']}")
                st.write(f"- Errors: {results['path_update']['errors']}")

                # Status update results
                st.write("**Status Updates:**")
                st.write(f"- Updated: {results['status_update']['updated']}")
                st.write(f"- Errors: {results['status_update']['errors']}")

                # Total filings
                st.write(f"**Total Filings:** {results['total_filings']}")

                # Add a button to refresh the page
                if st.button("Refresh Page"):
                    st.rerun()

# Inventory Summary tab
with data_mgmt_tabs[1]:
    st.subheader("Inventory Summary")
    st.markdown("""
    View a summary of your filing inventory, including counts by company, filing type, and processing status.
    """)

    # Get inventory summary
    summary = get_inventory_summary()

    if summary and "error" not in summary:
        # Display total filings
        st.metric("Total Filings", summary.get("total_filings", 0))

        # Display status counts
        st.subheader("Filing Status Counts")
        status_counts_df = pd.DataFrame(summary.get("status_counts", []))
        if not status_counts_df.empty and "processing_status" in status_counts_df.columns and "count" in status_counts_df.columns:
            st.bar_chart(status_counts_df.set_index("processing_status")["count"])
            st.dataframe(status_counts_df, use_container_width=True)
        else:
            st.info("No filing status counts available.")

        # Display company counts
        st.subheader("Filings by Company")
        company_counts_df = pd.DataFrame(summary.get("company_counts", []))
        if not company_counts_df.empty and "ticker" in company_counts_df.columns and "count" in company_counts_df.columns:
            # Display top 10 companies
            top_companies = company_counts_df.head(10)
            st.bar_chart(top_companies.set_index("ticker")["count"])
            st.dataframe(company_counts_df, use_container_width=True)
        else:
            st.info("No company counts available.")

        # Display filing type counts
        st.subheader("Filings by Type")
        type_counts_df = pd.DataFrame(summary.get("type_counts", []))
        if not type_counts_df.empty and "filing_type" in type_counts_df.columns and "count" in type_counts_df.columns:
            st.bar_chart(type_counts_df.set_index("filing_type")["count"])
            st.dataframe(type_counts_df, use_container_width=True)
        else:
            st.info("No filing type counts available.")

        # Display year counts
        st.subheader("Filings by Year")
        year_counts_df = pd.DataFrame(summary.get("year_counts", []))
        if not year_counts_df.empty and "year" in year_counts_df.columns and "count" in year_counts_df.columns:
            st.bar_chart(year_counts_df.set_index("year")["count"])
            st.dataframe(year_counts_df, use_container_width=True)
        else:
            st.info("No year counts available.")
    else:
        st.error(f"Error retrieving inventory summary: {summary.get('error', 'Unknown error')}")

# Recent ETL Jobs tab
with data_mgmt_tabs[2]:
    st.subheader("Recent ETL Jobs")
    st.markdown("""
    View recent ETL jobs and their status. You can also see the progress of currently running jobs.
    """)

    # Get jobs from the ETL service
    jobs = etl_service.get_jobs()

    # Display jobs
    if jobs:
        jobs_df = pd.DataFrame(jobs)
        st.dataframe(jobs_df, use_container_width=True)

        # Add a button to refresh the jobs list
        if st.button("Refresh Jobs"):
            st.rerun()

        # Show active job if any
        active_job = etl_service.get_active_job()
        if active_job:
            st.info(f"Job {active_job.job_id} is currently running. Progress: {active_job.progress}%")

            # Show progress bar
            st.progress(active_job.progress / 100)

            # Show current stage
            st.write(f"Current stage: {active_job.current_stage}")

            # Show logs
            st.subheader("Job Logs")
            logs = etl_service.get_job_logs(active_job.job_id)
            if logs:
                st.code("\n".join(logs[-20:]), language="bash")
            else:
                st.info("No logs available.")
    else:
        st.info("No ETL jobs have been run yet. Use the 'Run ETL Pipeline' button above to start a new job.")

# Footer
st.markdown("---")
st.info("This page is under development. More features will be added in future versions.")
