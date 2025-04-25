"""
Simplified SEC Filing Analyzer App

A simplified version of the SEC Filing Analyzer Streamlit app
that doesn't depend on the SEC Filing Analyzer package.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

# Set page config
st.set_page_config(
    page_title="SEC Filing Analyzer (Simplified)", page_icon="📊", layout="wide", initial_sidebar_state="expanded"
)

# Title and description
st.title("SEC Filing Analyzer (Simplified)")
st.markdown("""
This is a simplified version of the SEC Filing Analyzer application.
If you can see this page, Streamlit is working correctly!
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Dashboard", "ETL Pipeline", "Agent Workflow", "Data Explorer"])

# Page content
if page == "Dashboard":
    st.header("Dashboard")
    st.write("Welcome to the SEC Filing Analyzer dashboard.")

    # System status
    st.subheader("System Status")

    # Check if data directory exists
    data_dir = "data"
    data_exists = os.path.exists(data_dir)

    # Display status
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Data Directory", "Available" if data_exists else "Not Found")
        st.metric("Streamlit", "Running")

    with col2:
        st.metric("Python Version", f"{os.sys.version.split()[0]}")
        st.metric("Current Directory", os.path.basename(os.getcwd()))

    # Quick actions
    st.subheader("Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Sample ETL Process", use_container_width=True):
            st.info("This would start an ETL process in the full app.")

    with col2:
        if st.button("Sample Agent Query", use_container_width=True):
            st.info("This would start an agent query in the full app.")

    with col3:
        if st.button("Sample Data Exploration", use_container_width=True):
            st.info("This would open data exploration in the full app.")

elif page == "ETL Pipeline":
    st.header("ETL Pipeline")
    st.write("This is a simplified version of the ETL Pipeline page.")

    # Sample form
    with st.form("sample_etl_form"):
        st.subheader("Sample ETL Configuration")

        ticker = st.text_input("Ticker Symbol", "AAPL")
        filing_type = st.selectbox("Filing Type", ["10-K", "10-Q", "8-K"])
        process_data = st.checkbox("Process Data", value=True)

        submitted = st.form_submit_button("Run Sample ETL")

        if submitted:
            st.success("Sample ETL process would start in the full app.")

            # Sample progress
            import time

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Processing: {i + 1}%")
                time.sleep(0.01)

            status_text.text("Processing complete!")
            st.balloons()

elif page == "Agent Workflow":
    st.header("Agent Workflow")
    st.write("This is a simplified version of the Agent Workflow page.")

    # Sample chat interface
    st.subheader("Sample Chat Interface")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm a simplified version of the SEC Filing Analyzer agent. How can I help you today?",
            }
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for new message
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Simulate agent response
        with st.chat_message("assistant"):
            response = f"This is a simplified demo. In the full app, I would analyze your query: '{user_input}' and provide financial insights."
            st.markdown(response)

            # Add assistant message to chat
            st.session_state.messages.append({"role": "assistant", "content": response})

elif page == "Data Explorer":
    st.header("Data Explorer")
    st.write("This is a simplified version of the Data Explorer page.")

    # Sample data
    st.subheader("Sample Financial Data")

    data = {
        "Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
        "Revenue": [1000000, 1100000, 1050000, 1200000, 1300000],
        "Expenses": [800000, 850000, 900000, 950000, 1000000],
        "Profit": [200000, 250000, 150000, 250000, 300000],
    }

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # Sample visualization
    st.subheader("Sample Visualization")

    chart_type = st.selectbox("Chart Type", ["Line", "Bar"])

    if chart_type == "Line":
        st.line_chart(df.set_index("Date")[["Revenue", "Expenses", "Profit"]])
    else:
        st.bar_chart(df.set_index("Date")[["Revenue", "Expenses", "Profit"]])

    # Sample query
    st.subheader("Sample Query")

    query = st.text_area("SQL Query", "SELECT * FROM financial_data WHERE Date > '2023-02-01'")

    if st.button("Run Query"):
        st.info("This would execute the query in the full app.")
        filtered_df = df[df["Date"] > "2023-02-01"]
        st.dataframe(filtered_df, use_container_width=True)
