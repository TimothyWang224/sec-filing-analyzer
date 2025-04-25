"""
Minimal Streamlit App

This is a minimal Streamlit app to test if Streamlit is working correctly.
"""

import streamlit as st

st.title("Minimal Streamlit App")
st.write("If you can see this, Streamlit is working correctly!")

# Add a simple widget
number = st.slider("Select a number", 0, 100, 50)
st.write(f"You selected: {number}")

# Display some information about the environment
st.subheader("Environment Information")
import sys

st.write(f"Python version: {sys.version}")
st.write(f"Streamlit version: {st.__version__}")
