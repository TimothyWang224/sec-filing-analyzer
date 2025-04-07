"""
Test script for the SEC Filing ETL Pipeline
"""

import asyncio
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    # Check for required environment variables
    edgar_identity = os.getenv("EDGAR_IDENTITY")
    if not edgar_identity:
        logger.error("EDGAR_IDENTITY environment variable not set. Please set it in your .env file.")
        return
    
    logger.info(f"Using EDGAR identity: {edgar_identity}")
    
    # Initialize pipeline with Neo4j if credentials are available
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    use_neo4j = all([neo4j_uri, neo4j_user, neo4j_password])
    if use_neo4j:
        logger.info("Neo4j credentials found. Will store data in Neo4j database.")
    else:
        logger.info("No Neo4j credentials found. Will use in-memory storage.")
    
    pipeline = SECFilingETLPipeline(
        use_neo4j=use_neo4j,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password
    )
    
    # Test parameters
    ticker = "AAPL"
    current_year = datetime.now().year
    years = [current_year - 1]  # Use previous year to ensure we have complete filings
    
    try:
        logger.info(f"Starting ETL process for {ticker} for years {years}")
        
        # Process company filings
        results = await pipeline.process_company(ticker, years)
        
        # Print results
        if results["filings"]:
            logger.info(f"Successfully processed {len(results['filings'])} filings for {ticker}")
            for filing in results["filings"]:
                logger.info(f"\nProcessed {filing['form_type']} from {filing['filing_date']}")
                
                # Log XBRL data
                if filing["xbrl_data"]:
                    logger.info("XBRL data extracted:")
                    for metric, value in filing["xbrl_data"].items():
                        logger.info(f"  {metric}: {value}")
                
                # Log structure data
                if filing["structure"]:
                    logger.info("Filing structure extracted:")
                    if "sections" in filing["structure"]:
                        logger.info(f"  Found {len(filing['structure']['sections'])} sections")
                    if "metadata" in filing["structure"]:
                        logger.info("  Metadata extracted")
                
                # Log entity data
                if filing["entities"]:
                    logger.info("Entities extracted:")
                    for entity_type, entities in filing["entities"].items():
                        logger.info(f"  {entity_type}: {len(entities)} entities found")
        else:
            logger.warning(f"No filings processed for {ticker}")
            
        # Print any errors
        if results["errors"]:
            logger.warning("Encountered errors:")
            for error in results["errors"]:
                logger.error(f"- {error['error']}")
                
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    asyncio.run(main()) 