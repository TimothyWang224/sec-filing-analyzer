"""
SEC ETL Tool

Tool for agents to interact with the SEC filing ETL pipeline.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from rich.console import Console

from ..sec_filing_analyzer.etl.pipeline import ETLPipeline
from ..sec_filing_analyzer.etl.config import ETLConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class SECETLTool:
    """Tool for agents to interact with the SEC filing ETL pipeline."""
    
    def __init__(self, config: Optional[ETLConfig] = None):
        """Initialize the tool.
        
        Args:
            config: Optional ETL configuration
        """
        self.config = config or ETLConfig.from_env()
        self.pipeline = ETLPipeline(
            cache_dir=self.config.cache_dir,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            embedding_model=self.config.embedding_model
        )
    
    def process_company(
        self,
        ticker: str,
        years: Optional[List[int]] = None,
        filing_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process SEC filings for a company.
        
        Args:
            ticker: Company ticker symbol
            years: Optional list of years to process
            filing_types: Optional list of filing types to process
            
        Returns:
            Dictionary containing processed filings and metadata
        """
        try:
            return self.pipeline.process_company(
                ticker=ticker,
                years=years,
                filing_types=filing_types,
                show_progress=True
            )
        except Exception as e:
            logger.error(f"Error processing company {ticker}: {str(e)}")
            raise
    
    def process_companies(
        self,
        tickers: List[str],
        years: Optional[List[int]] = None,
        filing_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process SEC filings for multiple companies.
        
        Args:
            tickers: List of company ticker symbols
            years: Optional list of years to process
            filing_types: Optional list of filing types to process
            
        Returns:
            Dictionary containing results for each company
        """
        try:
            return self.pipeline.process_companies(
                tickers=tickers,
                years=years,
                filing_types=filing_types,
                show_progress=True
            )
        except Exception as e:
            logger.error(f"Error processing companies: {str(e)}")
            raise
    
    def get_company_metadata(self, ticker: str) -> Dict[str, Any]:
        """Get metadata for a company.
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Dictionary containing company metadata
        """
        try:
            return self.pipeline.get_company_metadata(ticker)
        except Exception as e:
            logger.error(f"Error getting metadata for {ticker}: {str(e)}")
            raise
    
    def check_company_exists(self, ticker: str) -> bool:
        """Check if a company exists.
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Boolean indicating if company exists
        """
        try:
            return self.pipeline.check_company_exists(ticker)
        except Exception as e:
            logger.error(f"Error checking company {ticker}: {str(e)}")
            raise 