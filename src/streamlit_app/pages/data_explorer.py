"""
Data Explorer Page

This page provides a user interface for exploring the data extracted from SEC filings.
"""

import logging
import os
import sys
import traceback
from pathlib import Path

import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("data_explorer_debug.log")],
)
logger = logging.getLogger("data_explorer")

# Log startup information
logger.info("Data Explorer starting up")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Import utility functions

# Import data components
try:
    logger.info("Importing SEC Filing Analyzer components")
    from sec_filing_analyzer.config import ConfigProvider, StorageConfig

    # Import LlamaIndexVectorStore only if needed
    config_imports_successful = True
    logger.info("Successfully imported SEC Filing Analyzer components")
except ImportError as e:
    error_msg = f"Error importing SEC Filing Analyzer components: {e}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())
    st.error(error_msg)
    st.warning(
        "Some functionality may be limited. Please make sure the SEC Filing Analyzer package is installed correctly."
    )
    config_imports_successful = False

    # Define fallback classes for when imports fail
    class FallbackConfig:
        def __init__(self):
            self.vector_store_path = "data/vector_store"

    class FallbackConfigProvider:
        @staticmethod
        def initialize():
            pass

        @staticmethod
        def get_config(*args, **kwargs):
            # Ignore parameters
            _ = args, kwargs  # Suppress unused variable warning
            return FallbackConfig()


# Set page config
st.set_page_config(
    page_title="Data Explorer - SEC Filing Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize configuration
logger.info("Initializing configuration")
if config_imports_successful:
    try:
        logger.info("Calling ConfigProvider.initialize()")
        ConfigProvider.initialize()
        logger.info("Getting storage configuration")
        storage_config = ConfigProvider.get_config(StorageConfig)
        logger.info(f"Successfully initialized configuration: {storage_config}")
    except Exception as config_error:
        error_msg = f"Error initializing configuration: {config_error}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(error_msg)
        logger.info("Using fallback configuration")
        storage_config = FallbackConfig()
else:
    # Use fallback configuration
    logger.info("Using fallback configuration due to import failure")
    storage_config = FallbackConfig()

# Title and description
st.title("Data Explorer")
st.markdown("""
Explore the data extracted from SEC filings, including:
- Semantic data in the vector store
- Graph relationships

Note: For exploring quantitative data in DuckDB, please use the DuckDB UI button on the main dashboard.
""")

# Sidebar for navigation
st.sidebar.header("Explorer Navigation")
logger.info("Setting up sidebar navigation")
explorer_type = st.sidebar.radio("Select Explorer", ["Semantic Search", "Graph Explorer"])
logger.info(f"Selected explorer type: {explorer_type}")

# Main content
if explorer_type == "Semantic Search":
    st.header("Semantic Search")
    logger.info("Entering Semantic Search section")

    # Check if imports were successful
    if not config_imports_successful:
        logger.warning("Semantic Search requires the SEC Filing Analyzer package")
        st.error("Semantic Search requires the SEC Filing Analyzer package.")
        st.info("Please make sure the package is installed correctly using 'poetry install'.")
    else:
        # Vector store path
        logger.info(f"Using vector store path: {storage_config.vector_store_path}")
        vector_store_path = storage_config.vector_store_path

        if not os.path.exists(vector_store_path):
            logger.warning(f"Vector store not found at {vector_store_path}")
            st.warning(f"Vector store not found at {vector_store_path}. Please run the ETL pipeline first.")
        else:
            logger.info(f"Vector store found at {vector_store_path}")
            # Initialize vector store
            try:
                logger.info("Using demo mode instead of actual vector store initialization")
                # Use a placeholder instead of actual initialization to avoid errors
                # vector_store = LlamaIndexVectorStore(store_path=vector_store_path)
                st.success("Vector store found. Using demo mode for now.")

                # Semantic search
                st.subheader("Semantic Search")

                # Company filter
                companies = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
                selected_company = st.selectbox("Filter by Company", ["All"] + companies)

                # Search query
                query = st.text_input("Search Query", "")

                if st.button("Search") and query:
                    with st.spinner("Searching..."):
                        try:
                            # Placeholder search functionality
                            st.info("Search functionality will be implemented here.")

                            # Placeholder results
                            results = [
                                {
                                    "text": "This is a sample result that would come from the vector store.",
                                    "metadata": {
                                        "company": "AAPL",
                                        "filing_type": "10-K",
                                        "filing_date": "2023-01-01",
                                        "section": "Risk Factors",
                                    },
                                    "score": 0.95,
                                },
                                {
                                    "text": "Another sample result with different metadata.",
                                    "metadata": {
                                        "company": "MSFT",
                                        "filing_type": "10-Q",
                                        "filing_date": "2023-03-31",
                                        "section": "Management Discussion",
                                    },
                                    "score": 0.85,
                                },
                            ]

                            # Display results
                            for i, result in enumerate(results):
                                st.subheader(f"Result {i + 1} (Score: {result['score']:.2f})")
                                st.write(f"**Company:** {result['metadata']['company']}")
                                st.write(
                                    f"**Filing:** {result['metadata']['filing_type']} ({result['metadata']['filing_date']})"
                                )
                                st.write(f"**Section:** {result['metadata']['section']}")
                                st.text_area(f"Text {i + 1}", result["text"], height=150)
                        except Exception as e:
                            st.error(f"Error performing search: {str(e)}")

                # Vector store statistics
                st.subheader("Vector Store Statistics")

                # Placeholder statistics
                stats = {
                    "Total Documents": "1,245",
                    "Total Chunks": "15,678",
                    "Companies": "25",
                    "Filing Types": "10-K, 10-Q, 8-K",
                    "Date Range": "2020-01-01 to 2023-12-31",
                    "Embedding Model": "text-embedding-3-small",
                }

                # Display statistics
                col1, col2 = st.columns(2)

                with col1:
                    for key in list(stats.keys())[:3]:
                        st.metric(key, stats[key])

                with col2:
                    for key in list(stats.keys())[3:]:
                        st.metric(key, stats[key])

            except Exception as e:
                error_msg = f"Error in Vector Store Explorer: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(error_msg)
                st.info("Using demo mode instead.")

                # Fallback to demo mode
                logger.info("Falling back to demo mode")
                st.subheader("Demo Mode")
                st.warning("Vector store functionality is currently in demo mode.")

                # Show some sample data
                st.subheader("Sample Data")
                st.write("This is sample data that would normally come from the vector store.")

elif explorer_type == "Graph Explorer":
    st.header("Graph Explorer")

    # Placeholder for graph explorer
    st.info("Graph Explorer will be implemented in a future version.")

    # Neo4j connection details
    st.subheader("Neo4j Connection")

    col1, col2 = st.columns(2)

    with col1:
        neo4j_url = st.text_input("Neo4j URL", "bolt://localhost:7687")

    with col2:
        neo4j_database = st.text_input("Neo4j Database", "neo4j")

    # Username and password
    col1, col2 = st.columns(2)

    with col1:
        neo4j_username = st.text_input("Username", "neo4j")

    with col2:
        neo4j_password = st.text_input("Password", "", type="password")

    if st.button("Connect to Neo4j"):
        st.info("Neo4j connection functionality will be implemented in a future version.")

    # Graph visualization placeholder
    st.subheader("Graph Visualization")

    st.image(
        "https://neo4j.com/wp-content/uploads/graph-example.png",
        caption="Sample Graph Visualization (Placeholder)",
    )

    # Cypher query
    st.subheader("Cypher Query")

    query = st.text_area("Cypher Query", "MATCH (n) RETURN n LIMIT 10")

    if st.button("Run Query"):
        st.info("Cypher query execution will be implemented in a future version.")
