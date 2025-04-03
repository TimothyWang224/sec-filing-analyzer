"""
SEC Filing ETL Pipeline

Main pipeline class that integrates the company-deep-research ETL functionality.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add company-deep-research to Python path
project_root = Path(__file__).parent.parent.parent.parent
external_libs = project_root / "external_libs"
company_deep_research = external_libs / "company-deep-research"
sys.path.insert(0, str(company_deep_research))

# Import from company-deep-research
from data_retrieval.retrieve_sec_data import SECFilingsDownloader
from graphrag.llamaindex_integration import LlamaIndexIntegration
from graphrag.store import GraphStore
from graphrag.sec_structure import SECStructure
from graphrag.sec_entities import SECEntities

from .config import ETLConfig
from ..data_retrieval.filing_processor import FilingProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class SECETLPipeline:
    """Main ETL pipeline for processing SEC filings."""
    
    def __init__(self, config: Optional[ETLConfig] = None):
        """Initialize the ETL pipeline.
        
        Args:
            config: Optional configuration. If not provided, will use defaults.
        """
        self.config = config or ETLConfig()
        
        # Create output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.downloader = SECFilingsDownloader(self.config.cache_dir)
        self.processor = FilingProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            embedding_model=self.config.embedding_model
        )
        
        # Initialize graph store if Neo4j is configured
        if all([self.config.neo4j_uri, self.config.neo4j_user, self.config.neo4j_password]):
            self.graph_store = GraphStore(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password
            )
        else:
            self.graph_store = None
            console.print("[yellow]Warning: Neo4j not configured. Graph features will be disabled.[/yellow]")
        
        # Initialize processing components
        self.sec_structure = SECStructure()
        self.sec_entities = SECEntities()
        
        console.print("[green]Initialized SEC ETL Pipeline[/green]")
    
    async def process_company(
        self,
        ticker: str,
        years: Optional[List[int]] = None,
        filing_types: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Process SEC filings for a company.
        
        Args:
            ticker: Company ticker symbol
            years: Optional list of years to process. If None, processes most recent year.
            filing_types: Optional list of filing types to process.
            show_progress: Whether to show progress bars
            
        Returns:
            Dict containing processing results and metadata
        """
        filing_types = filing_types or self.config.filing_types
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not show_progress
        ) as progress:
            # Step 1: Download filings
            task = progress.add_task(
                f"Downloading filings for {ticker}...",
                total=None
            )
            filings = self.downloader.get_filings(
                ticker=ticker,
                years=years,
                filing_types=filing_types
            )
            progress.update(task, completed=True)
            
            # Step 2: Process documents
            process_task = progress.add_task(
                f"Processing {len(filings)} filings...",
                total=len(filings)
            )
            processed_docs = []
            for filing in filings:
                # Extract structure
                structure = self.sec_structure.analyze(filing)
                
                # Extract entities
                entities = self.sec_entities.extract(filing)
                
                # Create document chunks
                chunks = self.processor.process_filing(filing)
                
                processed_docs.append({
                    "filing": filing,
                    "structure": structure,
                    "entities": entities,
                    "chunks": chunks
                })
                self.processor.store_in_vector_db(processed_docs[-1])
                progress.advance(process_task)
            
            # Step 3: Build graph if Neo4j is configured
            if self.graph_store:
                graph_task = progress.add_task("Building knowledge graph...", total=len(processed_docs))
                for doc in processed_docs:
                    await self.graph_store.add_document(
                        doc["filing"],
                        doc["structure"],
                        doc["entities"]
                    )
                    progress.advance(graph_task)
        
        return {
            "ticker": ticker,
            "years": years,
            "filing_types": filing_types,
            "num_filings": len(filings),
            "num_chunks": sum(len(doc["chunks"]) for doc in processed_docs),
            "graph_created": self.graph_store is not None
        }
    
    async def check_company_exists(self, ticker: str) -> bool:
        """Check if company data exists in the vector database.
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            bool indicating if company exists
        """
        return await self.processor.check_company_exists(ticker)
    
    async def get_company_metadata(self, ticker: str) -> Dict[str, Any]:
        """Get metadata about processed company data.
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Dict containing metadata
        """
        return await self.processor.get_company_metadata(ticker)
    
    def process_companies(
        self,
        tickers: List[str],
        years: Optional[List[int]] = None,
        filing_types: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Process filings for multiple companies.
        
        Args:
            tickers: List of company ticker symbols
            years: List of years to process
            filing_types: List of filing types to process
            show_progress: Whether to show progress bars
            
        Returns:
            Dictionary containing results for each company
        """
        results = {}
        
        for ticker in tickers:
            console.print(f"\n[bold blue]Processing {ticker}...[/bold blue]")
            results[ticker] = self.process_company(
                ticker=ticker,
                years=years,
                filing_types=filing_types,
                show_progress=show_progress
            )
        
        return results 