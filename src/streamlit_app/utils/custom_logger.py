"""
Custom Logger for Streamlit App

This module provides a custom logger that captures output to the terminal component.
"""

import logging
import sys
from typing import Optional

# Import the terminal output capture
from src.streamlit_app.components.terminal_output import TerminalOutputCapture


class StreamlitLogHandler(logging.Handler):
    """Custom log handler that sends log messages to the Streamlit terminal component."""

    def emit(self, record):
        """Emit a log record to the Streamlit terminal component."""
        try:
            msg = self.format(record)
            TerminalOutputCapture.add_output(msg)
        except Exception:
            self.handleError(record)


def setup_custom_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a custom logger that sends output to both the console and the Streamlit terminal component.

    Args:
        name: Name of the logger
        level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create a Streamlit handler
    streamlit_handler = StreamlitLogHandler()
    streamlit_handler.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    streamlit_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(streamlit_handler)

    return logger
