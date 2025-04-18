"""
Utilities for the Streamlit app.
"""

# Import utility functions
try:
    from .duckdb_ui import launch_duckdb_ui
except ImportError:
    # Fallback function if the module is missing
    def launch_duckdb_ui(db_path=None):
        import streamlit as st
        st.error("DuckDB UI launcher is not available.")
        return None

# Import app state module
from . import app_state

__all__ = [
    "launch_duckdb_ui",
    "app_state"
]
