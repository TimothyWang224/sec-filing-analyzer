"""
SEC Filing Analyzer - Streamlit Application

Main entry point for the SEC Filing Analyzer Streamlit application.
"""

import streamlit as st

# Set page config first (must be the first Streamlit command)
st.set_page_config(
    page_title="SEC Filing Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import terminal output component
from src.streamlit_app.components.terminal_output import TerminalOutputCapture

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sec_filing_analyzer_app")

# Start capturing terminal output
TerminalOutputCapture.start_capture()

# Import configuration
from sec_filing_analyzer.config import ConfigProvider, StreamlitConfig

# Import utility functions
from src.streamlit_app.utils import launch_duckdb_ui

# Page config is already set at the top of the file

# Initialize configuration
ConfigProvider.initialize()
streamlit_config = ConfigProvider.get_config(StreamlitConfig)

# Title and description
st.title("SEC Filing Analyzer")
st.markdown("""
This application provides a user interface for the SEC Filing Analyzer,
allowing you to extract, analyze, and explore SEC filings data using
both ETL pipelines and intelligent agents.
""")

# Sidebar content
st.sidebar.title("Dashboard")

# Add some useful information to the sidebar
st.sidebar.info("""
**Welcome to SEC Filing Analyzer!**

Use the navigation menu at the top of the sidebar to access different pages:
- **app**: This dashboard
- **etl_data_inventory**: View companies and filings in the system (Primary)
- **etl_pipeline**: Run data extraction pipelines
- **agent_workflow**: Interact with intelligent agents
- **data_explorer**: Explore extracted data
- **data_management**: Manage data across storage systems
- **configuration**: Configure system settings
""")

# Main dashboard content
st.header("Dashboard")
st.write("Welcome to the SEC Filing Analyzer dashboard.")

# Add a terminal output component to the sidebar
with st.sidebar.expander("Terminal Output", expanded=False):
    # Import display_terminal_output
    from src.streamlit_app.components.terminal_output import display_terminal_output

    # Display the terminal output
    display_terminal_output("Global Terminal Output", height=400)

# System status
st.subheader("System Status")

# Check if DuckDB database exists
db_path = "data/db_backup/improved_financial_data.duckdb"
db_exists = os.path.exists(db_path)

# Check if vector store exists
vector_store_path = "data/vector_store"
vector_store_exists = os.path.exists(vector_store_path)

# Display status
col1, col2 = st.columns(2)

with col1:
    st.metric("DuckDB Database", "Available" if db_exists else "Not Found")
    st.metric("Vector Store", "Available" if vector_store_exists else "Not Found")

with col2:
    st.metric("ETL Pipeline", "Ready")
    st.metric("Agent System", "Ready")

# Quick actions
st.subheader("Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("ETL Data Inventory", use_container_width=True):
        # Use Streamlit's built-in page navigation
        st.switch_page("pages/etl_data_inventory.py")

with col2:
    if st.button("Run ETL Pipeline", use_container_width=True):
        # Use Streamlit's built-in page navigation
        st.switch_page("pages/etl_pipeline.py")

with col3:
    if st.button("Start Agent Workflow", use_container_width=True):
        # Use Streamlit's built-in page navigation
        st.switch_page("pages/agent_workflow.py")

with col4:
    if st.button("Open DuckDB UI", use_container_width=True):
        # Launch DuckDB UI using the helper function
        launch_duckdb_ui()

# Exit application
st.subheader("Application Control")

if st.button("Exit Application", type="primary", use_container_width=True):
    # Create a shutdown signal file
    shutdown_file = Path("shutdown_signal.txt")
    with open(shutdown_file, "w") as f:
        f.write("shutdown")
    st.success("Shutdown signal sent. The application will close shortly.")
    st.info("You can close this browser tab now.")
