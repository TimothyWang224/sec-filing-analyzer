"""
Script to run ETL process for SEC filings
"""

import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import os

from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline
from sec_filing_analyzer.config import ETLConfig, StorageConfig, Neo4jConfig
from sec_filing_analyzer.storage.graph_store import GraphStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_neo4j_config():
    """Get Neo4j configuration from environment variables or defaults."""
    config = Neo4jConfig()
    return {
        'url': os.getenv('NEO4J_URL') or os.getenv('NEO4J_URI') or config.url,
        'username': os.getenv('NEO4J_USERNAME') or os.getenv('NEO4J_USER') or config.username,
        'password': os.getenv('NEO4J_PASSWORD') or config.password,
        'database': os.getenv('NEO4J_DATABASE') or config.database
    }

def parse_args():
    neo4j_config = get_neo4j_config()
    parser = argparse.ArgumentParser(description='Process SEC filings for a company')
    parser.add_argument('ticker', help='Company ticker symbol (e.g., NVDA)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)', required=True)
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)', required=True)
    parser.add_argument('--filing-types', nargs='+', 
                       help='List of filing types to process (e.g., 10-K 10-Q)',
                       default=['10-K', '10-Q'])
    parser.add_argument('--use-neo4j', action='store_true',
                       help='Use Neo4j as the graph store')
    parser.add_argument('--neo4j-url', help='Neo4j server URL',
                       default=neo4j_config['url'])
    parser.add_argument('--neo4j-username', help='Neo4j username',
                       default=neo4j_config['username'])
    parser.add_argument('--neo4j-password', help='Neo4j password',
                       default=neo4j_config['password'])
    parser.add_argument('--neo4j-database', help='Neo4j database name',
                       default=neo4j_config['database'])
    return parser.parse_args()

def validate_dates(start_date: str, end_date: str) -> None:
    """Validate date formats and ranges."""
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if end < start:
            raise ValueError("End date must be after start date")
            
        if end > datetime.now():
            raise ValueError("End date cannot be in the future")
            
    except ValueError as e:
        if "time data" in str(e):
            raise ValueError("Dates must be in YYYY-MM-DD format")
        raise

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Validate dates
    validate_dates(args.start_date, args.end_date)
    
    # Initialize graph store with Neo4j if requested
    graph_store = None
    if args.use_neo4j:
        logger.info("Initializing Neo4j graph store...")
        graph_store = GraphStore(
            use_neo4j=True,
            username=args.neo4j_username,
            password=args.neo4j_password,
            url=args.neo4j_url,
            database=args.neo4j_database
        )
    
    # Initialize pipeline
    pipeline = SECFilingETLPipeline(graph_store=graph_store)
    
    try:
        logger.info(f"Starting ETL process for {args.ticker}")
        logger.info(f"Date range: {args.start_date} to {args.end_date}")
        logger.info(f"Filing types: {', '.join(args.filing_types)}")
        logger.info(f"Graph store: {'Neo4j' if args.use_neo4j else 'In-memory'}")
        
        # Process company filings
        pipeline.process_company(
            ticker=args.ticker,
            filing_types=args.filing_types,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        logger.info("ETL process completed successfully")
        
    except Exception as e:
        logger.error(f"Error running ETL process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 