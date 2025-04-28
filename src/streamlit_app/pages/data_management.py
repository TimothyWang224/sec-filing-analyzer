"""
Data Management Page

This page provides tools for managing data across different storage systems.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Import the lifecycle manager
from src.sec_filing_analyzer.config import ConfigProvider, ETLConfig, StorageConfig
from src.sec_filing_analyzer.storage.lifecycle_manager import DataLifecycleManager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("data_management.log")],
)
logger = logging.getLogger("data_management")

# Log startup information
logger.info("Data Management page starting up")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")

# Initialize configuration
ConfigProvider.initialize()
storage_config = ConfigProvider.get_config(StorageConfig)
etl_config = ConfigProvider.get_config(ETLConfig)

# Initialize lifecycle manager
lifecycle_manager = DataLifecycleManager(
    db_path=etl_config.db_path,  # Use db_path from ETLConfig
    vector_store_path=storage_config.vector_store_path,
    filings_dir=etl_config.filings_dir,
)

# Set page config
st.set_page_config(
    page_title="Data Management",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("Data Management")
st.markdown("""
This page provides tools for managing data across different storage systems.
You can view, delete, and manage filings across DuckDB, vector store, and file system.
""")

# Create tabs for different management tasks
tabs = st.tabs(["Filing Explorer", "Delete Filings", "Storage Stats"])

# Filing Explorer tab
with tabs[0]:
    st.header("Filing Explorer")
    st.markdown("""
    Explore filings across different storage systems.
    Select a company, filing type, and filing to view details.
    """)

    # Company selection
    companies_query = """
    SELECT ticker, name as company_name
    FROM companies
    ORDER BY ticker
    """
    companies_df = lifecycle_manager.conn.execute(companies_query).fetchdf()

    if not companies_df.empty:
        # Create a formatted list for the selectbox
        company_options = [f"{row['ticker']} - {row['company_name']}" for _, row in companies_df.iterrows()]

        selected_company = st.selectbox("Select Company", options=company_options)

        # Extract ticker from selection
        selected_ticker = selected_company.split(" - ")[0]

        # Get filing types for the selected company
        filing_types_result = lifecycle_manager.get_filing_types(selected_ticker)

        if "error" in filing_types_result:
            st.error(f"Error getting filing types: {filing_types_result['error']}")
        elif not filing_types_result["filing_types"]:
            st.info(f"No filings found for {selected_ticker}")
        else:
            # Create a selectbox for filing types
            filing_type_options = [f"{ft['filing_type']} ({ft['count']})" for ft in filing_types_result["filing_types"]]

            selected_filing_type_option = st.selectbox("Select Filing Type", options=filing_type_options)

            # Extract filing type from selection
            selected_filing_type = selected_filing_type_option.split(" (")[0]

            # Get filing dates for the selected company and filing type
            filing_dates_result = lifecycle_manager.get_filing_dates(selected_ticker, selected_filing_type)

            if "error" in filing_dates_result:
                st.error(f"Error getting filing dates: {filing_dates_result['error']}")
            elif not filing_dates_result["filing_dates"]:
                st.info(f"No {selected_filing_type} filings found for {selected_ticker}")
            else:
                # Create a selectbox for filing dates
                filing_date_options = [
                    f"{fd['filing_date']} - {fd['accession_number']}" for fd in filing_dates_result["filing_dates"]
                ]

                selected_filing_date_option = st.selectbox("Select Filing", options=filing_date_options)

                # Extract accession number from selection
                selected_accession_number = selected_filing_date_option.split(" - ")[1]

                # Get filing info
                filing_info = lifecycle_manager.get_filing_info(selected_accession_number)

                if "error" in filing_info:
                    st.error(f"Error getting filing info: {filing_info['error']}")
                else:
                    # Display filing info
                    st.subheader("Filing Information")

                    # Create columns for basic info
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Ticker", filing_info["ticker"])

                    with col2:
                        st.metric("Filing Type", filing_info["filing_type"])

                    with col3:
                        st.metric("Filing Date", filing_info["filing_date"])

                    # Create columns for storage info
                    col4, col5, col6 = st.columns(3)

                    with col4:
                        st.metric("Processing Status", filing_info["processing_status"])

                    with col5:
                        vector_store_info = filing_info["vector_store"]
                        has_embeddings = (
                            vector_store_info["document_embedding_exists"]
                            or vector_store_info["chunk_embeddings_count"] > 0
                        )
                        st.metric("Vector Store", "✅" if has_embeddings else "❌")

                    with col6:
                        file_system_info = filing_info["file_system"]
                        has_files = any(
                            isinstance(info, dict) and info.get("exists", False)
                            for subdir, info in file_system_info.items()
                        )
                        st.metric("File System", "✅" if has_files else "❌")

                    # Display detailed info in expandable sections
                    with st.expander("DuckDB Information"):
                        st.json(
                            {
                                "id": filing_info["id"],
                                "ticker": filing_info["ticker"],
                                "accession_number": filing_info["accession_number"],
                                "filing_type": filing_info["filing_type"],
                                "filing_date": filing_info["filing_date"],
                                "document_url": filing_info["document_url"],
                                "local_file_path": filing_info["local_file_path"],
                                "processing_status": filing_info["processing_status"],
                                "last_updated": filing_info["last_updated"],
                            }
                        )

                    with st.expander("Vector Store Information"):
                        st.json(filing_info["vector_store"])

                    with st.expander("File System Information"):
                        st.json(filing_info["file_system"])
    else:
        st.warning("No companies found in the database.")

# Delete Filings tab
with tabs[1]:
    st.header("Delete Filings")
    st.markdown("""
    Delete filings from all storage systems.
    This will remove the filing from DuckDB, vector store, and file system.
    """)

    # Warning
    st.warning("""
    **Warning**: Deletion is permanent and cannot be undone.
    Make sure you have backups before deleting any data.
    """)

    # Company selection
    if not companies_df.empty:
        # Create a formatted list for the selectbox
        company_options = [f"{row['ticker']} - {row['company_name']}" for _, row in companies_df.iterrows()]

        selected_company = st.selectbox("Select Company", options=company_options, key="delete_company")

        # Extract ticker from selection
        selected_ticker = selected_company.split(" - ")[0]

        # Get filing types for the selected company
        filing_types_result = lifecycle_manager.get_filing_types(selected_ticker)

        if "error" in filing_types_result:
            st.error(f"Error getting filing types: {filing_types_result['error']}")
        elif not filing_types_result["filing_types"]:
            st.info(f"No filings found for {selected_ticker}")
        else:
            # Create a selectbox for filing types
            filing_type_options = [f"{ft['filing_type']} ({ft['count']})" for ft in filing_types_result["filing_types"]]

            selected_filing_type_option = st.selectbox(
                "Select Filing Type",
                options=filing_type_options,
                key="delete_filing_type",
            )

            # Extract filing type from selection
            selected_filing_type = selected_filing_type_option.split(" (")[0]

            # Get filing dates for the selected company and filing type
            filing_dates_result = lifecycle_manager.get_filing_dates(selected_ticker, selected_filing_type)

            if "error" in filing_dates_result:
                st.error(f"Error getting filing dates: {filing_dates_result['error']}")
            elif not filing_dates_result["filing_dates"]:
                st.info(f"No {selected_filing_type} filings found for {selected_ticker}")
            else:
                # Create a selectbox for filing dates
                filing_date_options = [
                    f"{fd['filing_date']} - {fd['accession_number']}" for fd in filing_dates_result["filing_dates"]
                ]

                selected_filing_date_option = st.selectbox(
                    "Select Filing",
                    options=filing_date_options,
                    key="delete_filing_date",
                )

                # Extract accession number from selection
                selected_accession_number = selected_filing_date_option.split(" - ")[1]

                # Get filing info
                filing_info = lifecycle_manager.get_filing_info(selected_accession_number)

                if "error" in filing_info:
                    st.error(f"Error getting filing info: {filing_info['error']}")
                else:
                    # Display filing info
                    st.subheader("Filing to Delete")

                    # Create columns for basic info
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Ticker", filing_info["ticker"])

                    with col2:
                        st.metric("Filing Type", filing_info["filing_type"])

                    with col3:
                        st.metric("Filing Date", filing_info["filing_date"])

                    # Dry run option
                    dry_run = st.checkbox(
                        "Dry Run (simulate deletion without actually deleting files)",
                        value=True,
                    )

                    # Confirmation
                    confirmation = st.text_input(
                        f"Type '{selected_ticker}' to confirm deletion",
                        key="delete_confirmation",
                    )

                    # Delete button
                    if st.button(
                        "Delete Filing",
                        type="primary",
                        disabled=(confirmation != selected_ticker),
                    ):
                        if confirmation == selected_ticker:
                            # Show spinner while deleting
                            with st.spinner("Deleting filing..."):
                                # Delete filing
                                deletion_result = lifecycle_manager.delete_filing(
                                    selected_accession_number, dry_run=dry_run
                                )

                                if "error" in deletion_result:
                                    st.error(f"Error deleting filing: {deletion_result['error']}")
                                else:
                                    # Display deletion result
                                    if dry_run:
                                        st.success("Dry run completed successfully. No files were actually deleted.")
                                    else:
                                        st.success("Filing deleted successfully.")

                                    # Display detailed results
                                    st.subheader("Deletion Results")

                                    # Create columns for results
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        duckdb_status = deletion_result["duckdb"]["status"]
                                        st.metric(
                                            "DuckDB",
                                            "✅" if duckdb_status == "success" else "❌",
                                            help=f"Status: {duckdb_status}",
                                        )

                                    with col2:
                                        vector_store_status = deletion_result["vector_store"]["status"]
                                        st.metric(
                                            "Vector Store",
                                            "✅" if vector_store_status == "success" else "❌",
                                            help=f"Status: {vector_store_status}",
                                        )

                                    with col3:
                                        file_system_status = deletion_result["file_system"]["status"]
                                        st.metric(
                                            "File System",
                                            "✅" if file_system_status == "success" else "❌",
                                            help=f"Status: {file_system_status}",
                                        )

                                    # Display detailed results in expandable sections
                                    with st.expander("DuckDB Results"):
                                        st.json(deletion_result["duckdb"])

                                    with st.expander("Vector Store Results"):
                                        st.json(deletion_result["vector_store"])

                                    with st.expander("File System Results"):
                                        st.json(deletion_result["file_system"])
                        else:
                            st.error(f"Confirmation text does not match '{selected_ticker}'")
    else:
        st.warning("No companies found in the database.")

# Storage Stats tab
with tabs[2]:
    st.header("Storage Statistics")
    st.markdown("""
    View statistics about your storage systems.
    """)

    # Create columns for storage stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("DuckDB Statistics")

        # Get DuckDB stats
        try:
            # Get company count
            company_count = lifecycle_manager.conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]

            # Get filing count
            filing_count = lifecycle_manager.conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]

            # Get filing count by type
            filing_type_counts = lifecycle_manager.conn.execute("""
                SELECT filing_type, COUNT(*) as count
                FROM filings
                GROUP BY filing_type
                ORDER BY count DESC
            """).fetchdf()

            # Get filing count by processing status
            status_counts = lifecycle_manager.conn.execute("""
                SELECT processing_status, COUNT(*) as count
                FROM filings
                GROUP BY processing_status
                ORDER BY count DESC
            """).fetchdf()

            # Display stats
            st.metric("Companies", company_count)
            st.metric("Filings", filing_count)

            # Display filing type counts
            st.subheader("Filing Types")
            st.dataframe(filing_type_counts, use_container_width=True)

            # Display status counts
            st.subheader("Processing Status")
            st.dataframe(status_counts, use_container_width=True)

        except Exception as e:
            st.error(f"Error getting DuckDB stats: {e}")

    with col2:
        st.subheader("Vector Store Statistics")

        # Get vector store stats
        try:
            vector_store_path = Path(storage_config.vector_store_path)

            # Check if vector store exists
            if not vector_store_path.exists():
                st.warning("Vector store not found.")
            else:
                # Get company directories
                company_dir = vector_store_path / "by_company"
                if not company_dir.exists():
                    st.warning("Company directory not found in vector store.")
                else:
                    # Get companies with embeddings
                    companies = [d.name for d in company_dir.iterdir() if d.is_dir()]

                    # Get embedding counts by company
                    embedding_counts = {}
                    total_embeddings = 0

                    for company in companies:
                        company_path = company_dir / company
                        embedding_files = list(company_path.glob("*.npy"))
                        embedding_counts[company] = len(embedding_files)
                        total_embeddings += len(embedding_files)

                    # Display stats
                    st.metric("Companies", len(companies))
                    st.metric("Total Embeddings", total_embeddings)

                    # Display embedding counts by company
                    st.subheader("Embeddings by Company")

                    # Convert to dataframe
                    embedding_counts_df = pd.DataFrame(
                        {
                            "company": list(embedding_counts.keys()),
                            "count": list(embedding_counts.values()),
                        }
                    ).sort_values(by="count", ascending=False)

                    st.dataframe(embedding_counts_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error getting vector store stats: {e}")

    with col3:
        st.subheader("File System Statistics")

        # Get file system stats
        try:
            filings_dir = Path(etl_config.filings_dir)

            # Check if filings directory exists
            if not filings_dir.exists():
                st.warning("Filings directory not found.")
            else:
                # Get subdirectories
                subdirs = [d for d in filings_dir.iterdir() if d.is_dir()]

                # Get file counts by subdirectory
                subdir_counts = {}
                total_files = 0

                for subdir in subdirs:
                    # Count files recursively
                    file_count = sum(1 for _ in subdir.glob("**/*") if _.is_file())
                    subdir_counts[subdir.name] = file_count
                    total_files += file_count

                # Get total size
                total_size = sum(f.stat().st_size for f in filings_dir.glob("**/*") if f.is_file())
                total_size_mb = total_size / (1024 * 1024)

                # Display stats
                st.metric("Total Files", total_files)
                st.metric("Total Size", f"{total_size_mb:.2f} MB")

                # Display file counts by subdirectory
                st.subheader("Files by Subdirectory")

                # Convert to dataframe
                subdir_counts_df = pd.DataFrame(
                    {
                        "subdirectory": list(subdir_counts.keys()),
                        "count": list(subdir_counts.values()),
                    }
                ).sort_values(by="count", ascending=False)

                st.dataframe(subdir_counts_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error getting file system stats: {e}")

# Footer
st.markdown("---")
st.info("This page is under development. More features will be added in future versions.")


# Close lifecycle manager when the app is closed
def on_close():
    lifecycle_manager.close()


# Register the on_close function to be called when the app is closed
st.experimental_set_query_params(on_close=on_close)
