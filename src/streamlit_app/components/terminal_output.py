"""
Terminal Output Component for Streamlit App

This component displays terminal output in the Streamlit UI.
"""

import queue
import threading
import time
from typing import Callable, List, Optional

import streamlit as st

# Global queue for terminal output
terminal_output_queue = queue.Queue()

# Global flag to control output capture
capture_enabled = False


class TerminalOutputCapture:
    """Capture and display terminal output in Streamlit."""

    @staticmethod
    def start_capture():
        """Start capturing terminal output."""
        global capture_enabled
        capture_enabled = True

    @staticmethod
    def stop_capture():
        """Stop capturing terminal output."""
        global capture_enabled
        capture_enabled = False

    @staticmethod
    def add_output(text: str):
        """Add output to the queue."""
        if capture_enabled:
            terminal_output_queue.put(text)

    @staticmethod
    def get_output() -> List[str]:
        """Get all available output from the queue."""
        output = []
        while not terminal_output_queue.empty():
            try:
                output.append(terminal_output_queue.get_nowait())
            except queue.Empty:
                break
        return output


def display_terminal_output(title: str = "Terminal Output", height: int = 300):
    """
    Display terminal output in Streamlit.

    Args:
        title: Title for the terminal output section
        height: Height of the terminal output area in pixels
    """
    # Create a container for the terminal output
    st.subheader(title)

    # Create a placeholder for the terminal output
    terminal_container = st.empty()

    # Initialize session state for terminal output if it doesn't exist
    if "terminal_output" not in st.session_state:
        st.session_state.terminal_output = []

    # Function to update terminal output
    def update_terminal_output():
        # Get new output
        new_output = TerminalOutputCapture.get_output()

        # Add new output to session state
        if new_output:
            st.session_state.terminal_output.extend(new_output)

            # Update the terminal output display
            terminal_container.code("\n".join(st.session_state.terminal_output), language="bash")

    # Create a button to clear the terminal output
    if st.button("Clear Terminal Output"):
        st.session_state.terminal_output = []
        terminal_container.code("", language="bash")

    # Initial display
    terminal_container.code("\n".join(st.session_state.terminal_output), language="bash")

    # Start a background thread to update the terminal output
    update_thread = threading.Thread(target=lambda: update_terminal_output())
    update_thread.daemon = True
    update_thread.start()

    return terminal_container


def run_with_output_capture(func: Callable, *args, **kwargs):
    """
    Run a function and capture its output.

    Args:
        func: Function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function
    """
    # Start capturing output
    TerminalOutputCapture.start_capture()

    try:
        # Run the function
        result = func(*args, **kwargs)
        return result
    finally:
        # Stop capturing output
        TerminalOutputCapture.stop_capture()
