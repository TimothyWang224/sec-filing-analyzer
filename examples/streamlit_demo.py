#!/usr/bin/env python
"""
Streamlit UI for the SEC Filing Analyzer demo.

This script provides a web interface for the SEC Filing Analyzer demo.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# Import the build_demo_agent function directly
# We need to use an absolute import to avoid conflicts with src/examples
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_chat_demo import build_demo_agent

# Set demo mode environment variable
os.environ["SFA_DEMO_MODE"] = "1"


def main():
    """Run the Streamlit demo."""
    # Set page config
    st.set_page_config(
        page_title="SEC Filing Analyzer Demo",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize the agent
    agent = build_demo_agent()

    # Title and description
    st.title("💬 SEC Filing Analyzer")
    st.markdown("""
    This is a demo of the SEC Filing Analyzer. Ask questions about SEC filings and financial data.

    **Sample Questions:**
    - What was NVDA's revenue in 2023?
    - Find information about Apple's AI strategy
    - What are the risk factors for Microsoft?
    """)

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask a question about SEC filings...")
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = agent.respond(prompt)
            st.markdown(reply)

            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
