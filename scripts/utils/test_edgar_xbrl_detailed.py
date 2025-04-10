"""
Detailed test script for the edgar library's XBRL capabilities.
"""

import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import edgar

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_edgar_xbrl_detailed():
    """Test the edgar library's XBRL capabilities in detail."""
    try:
        # Create output directory
        output_dir = Path("data/test_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set edgar identity from environment variables
        import os
        edgar_identity = os.getenv("EDGAR_IDENTITY")
        if edgar_identity:
            edgar.set_identity(edgar_identity)
            logger.info(f"Set edgar identity to: {edgar_identity}")
        
        # Get Microsoft entity
        logger.info("Getting Microsoft entity...")
        msft = edgar.get_entity("MSFT")
        logger.info(f"Found Microsoft entity with CIK: {msft.cik}")
        
        # Get a specific filing
        logger.info("Getting a specific filing...")
        accession_number = "0001564590-22-026876"  # Microsoft's 10-K from July 2022
        
        # Get all filings
        filings = msft.get_filings()
        logger.info(f"Retrieved {len(filings)} filings")
        
        # Find the filing with the matching accession number
        filing = None
        for f in filings:
            if f.accession_number == accession_number:
                filing = f
                break
        
        if not filing:
            logger.error(f"Filing with accession number {accession_number} not found")
            return None
        
        logger.info(f"Found filing: {filing.form} filed on {filing.filing_date}")
        logger.info(f"Filing URL: {filing.filing_url}")
        
        # Check if the filing has XBRL data
        logger.info("Checking if filing has XBRL data...")
        has_xbrl = hasattr(filing, 'is_xbrl') and filing.is_xbrl
        logger.info(f"Filing has XBRL data: {has_xbrl}")
        
        if has_xbrl:
            # Get XBRL data
            logger.info("Getting XBRL data...")
            xbrl_data = filing.xbrl()
            
            # Print detailed information about the XBRL data
            logger.info("XBRL data detailed information:")
            logger.info(f"  Type: {type(xbrl_data)}")
            
            # Check for common attributes and methods
            for attr in dir(xbrl_data):
                if not attr.startswith('_'):
                    logger.info(f"  Attribute/Method: {attr}")
                    try:
                        value = getattr(xbrl_data, attr)
                        if callable(value):
                            logger.info(f"    Type: Method")
                        else:
                            logger.info(f"    Type: Attribute, Value: {value}")
                    except Exception as e:
                        logger.info(f"    Error accessing: {e}")
            
            # Try to access instance data
            if hasattr(xbrl_data, 'instance'):
                logger.info("Examining instance data...")
                instance = xbrl_data.instance
                
                # Check for facts
                if hasattr(instance, 'facts'):
                    facts = instance.facts
                    logger.info(f"  Number of facts: {len(facts)}")
                    
                    # Print the first few facts
                    logger.info("  First few facts:")
                    for i, fact in enumerate(facts[:5]):
                        logger.info(f"    Fact {i+1}: {fact}")
                
                # Try to query facts
                if hasattr(instance, 'query_facts'):
                    logger.info("Querying facts...")
                    try:
                        # Query US-GAAP facts
                        us_gaap_facts = instance.query_facts(schema='us-gaap')
                        logger.info(f"  Number of US-GAAP facts: {len(us_gaap_facts)}")
                        
                        # Print the first few US-GAAP facts
                        logger.info("  First few US-GAAP facts:")
                        for i, (idx, fact) in enumerate(us_gaap_facts.iterrows()[:5]):
                            logger.info(f"    Fact {i+1}: {idx}, {fact}")
                    except Exception as e:
                        logger.error(f"Error querying facts: {e}")
            
            # Try to access financial statements
            for statement_type in ['balance_sheet', 'income_statement', 'cash_flow_statement']:
                method_name = f'get_{statement_type}'
                if hasattr(xbrl_data, method_name):
                    logger.info(f"Trying to get {statement_type}...")
                    try:
                        statement = getattr(xbrl_data, method_name)()
                        if statement:
                            logger.info(f"  Found {statement_type}")
                            logger.info(f"  Statement type: {type(statement)}")
                            
                            # Check for common statement attributes
                            for attr in ['name', 'label', 'entity', 'periods', 'data']:
                                if hasattr(statement, attr):
                                    value = getattr(statement, attr)
                                    logger.info(f"    {attr}: {value}")
                        else:
                            logger.info(f"  No {statement_type} found")
                    except Exception as e:
                        logger.error(f"Error getting {statement_type}: {e}")
            
            return xbrl_data
        else:
            logger.warning("Filing does not have XBRL data")
            return None
    
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_edgar_xbrl_detailed()
