"""
ETL Service Initialization Module

This module provides functions to initialize the ETL service with proper SEC identity.
"""

import logging
import os

from dotenv import load_dotenv
from edgar import set_identity

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_edgar_identity():
    """
    Initialize the edgar identity from environment variables.

    Returns:
        bool: True if identity was set successfully, False otherwise
    """
    # Check if EDGAR_IDENTITY is set in environment variables
    edgar_identity = os.getenv("EDGAR_IDENTITY")

    if edgar_identity:
        # Set the identity in the edgar package
        set_identity(edgar_identity)
        logger.info(f"Set edgar identity to: {edgar_identity}")
        return True
    else:
        logger.error("EDGAR_IDENTITY environment variable not set. Please set it in your .env file.")
        return False
