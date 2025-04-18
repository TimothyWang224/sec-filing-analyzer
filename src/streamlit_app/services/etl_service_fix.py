"""
ETL Service Fix for Streamlit Application

This module provides a fix for the ETL service initialization issue.
"""

import os
import logging
from edgar import set_identity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_etl_service():
    """
    Fix the ETL service by setting the SEC identity.
    
    This function should be called before initializing the ETL service.
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
