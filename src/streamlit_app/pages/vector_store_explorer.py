"""
Vector Store Explorer

A standalone version of the Vector Store Explorer with robust error handling.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Vector Store Explorer - SEC Filing Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Vector Store Explorer")
st.markdown("""
This is a standalone version of the Vector Store Explorer with robust error handling.
It's designed to help diagnose issues with the main Data Explorer.
""")

# Vector store path
vector_store_path = "data/vector_store"

# Check if vector store exists
if not os.path.exists(vector_store_path):
    st.warning(f"Vector store not found at {vector_store_path}. Please run the ETL pipeline first.")
else:
    st.success(f"Vector store found at {vector_store_path}")
    
    # List files in vector store
    files = list(Path(vector_store_path).glob("*"))
    if files:
        st.write(f"Found {len(files)} files in vector store:")
        for file in files:
            st.write(f"- {file.name}")
    else:
        st.warning("No files found in vector store")
    
    # Demo mode
    st.subheader("Demo Mode")
    st.info("Vector store functionality is currently in demo mode.")
    
    # Company filter
    companies = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    selected_company = st.selectbox("Filter by Company", ["All"] + companies)
    
    # Search query
    query = st.text_input("Search Query", "")
    
    if st.button("Search") and query:
        with st.spinner("Searching..."):
            # Placeholder results
            results = [
                {
                    "text": "This is a sample result that would come from the vector store.",
                    "metadata": {
                        "company": "AAPL",
                        "filing_type": "10-K",
                        "filing_date": "2023-01-01",
                        "section": "Risk Factors"
                    },
                    "score": 0.95
                },
                {
                    "text": "Another sample result with different metadata.",
                    "metadata": {
                        "company": "MSFT",
                        "filing_type": "10-Q",
                        "filing_date": "2023-03-31",
                        "section": "Management Discussion"
                    },
                    "score": 0.85
                }
            ]
            
            # Display results
            for i, result in enumerate(results):
                st.subheader(f"Result {i+1} (Score: {result['score']:.2f})")
                st.write(f"**Company:** {result['metadata']['company']}")
                st.write(f"**Filing:** {result['metadata']['filing_type']} ({result['metadata']['filing_date']})")
                st.write(f"**Section:** {result['metadata']['section']}")
                st.text_area(f"Text {i+1}", result['text'], height=150)
    
    # Vector store statistics
    st.subheader("Vector Store Statistics")
    
    # Placeholder statistics
    stats = {
        "Total Documents": "1,245",
        "Total Chunks": "15,678",
        "Companies": "25",
        "Filing Types": "10-K, 10-Q, 8-K",
        "Date Range": "2020-01-01 to 2023-12-31",
        "Embedding Model": "text-embedding-3-small"
    }
    
    # Display statistics
    col1, col2 = st.columns(2)
    
    with col1:
        for key in list(stats.keys())[:3]:
            st.metric(key, stats[key])
    
    with col2:
        for key in list(stats.keys())[3:]:
            st.metric(key, stats[key])
