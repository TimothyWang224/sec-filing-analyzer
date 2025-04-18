"""
Test Improved XBRL Extraction

This script tests the improved XBRL extraction capabilities.
"""

import sys
import os
import json
import logging
from pathlib import Path
import time
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sec_filing_analyzer.data_processing.parallel_xbrl_extractor import ParallelXBRLExtractor
from sec_filing_analyzer.storage.optimized_duckdb_store import OptimizedDuckDBStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_xbrl_extraction(ticker, accession_number):
    """Test improved XBRL extraction for a single filing.
    
    Args:
        ticker: Company ticker symbol
        accession_number: SEC accession number
    """
    # Initialize the extractor
    extractor = ParallelXBRLExtractor(
        cache_dir="data/xbrl_cache",
        max_workers=1,
        rate_limit=0.2
    )
    
    # Generate a filing ID
    filing_id = f"{ticker}_{accession_number.replace('-', '_')}"
    
    # Start timer
    start_time = time.time()
    
    # Extract financials
    financials = extractor.extract_financials(
        ticker=ticker,
        filing_id=filing_id,
        accession_number=accession_number
    )
    
    # End timer
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Print results
    print(f"\n=== XBRL Extraction Results for {ticker} {accession_number} ===")
    print(f"Extraction time: {elapsed:.2f} seconds")
    
    if "error" in financials:
        print(f"Error: {financials['error']}")
        return financials
    
    # Print basic information
    print(f"Filing Date: {financials.get('filing_date')}")
    print(f"Fiscal Year: {financials.get('fiscal_year')}")
    print(f"Fiscal Quarter: {financials.get('fiscal_quarter')}")
    print(f"Filing Type: {financials.get('filing_type')}")
    
    # Print facts
    facts = financials.get("facts", [])
    print(f"\nFacts: {len(facts)}")
    if facts:
        # Group facts by category
        facts_by_category = {}
        for fact in facts:
            category = fact.get("category", "unknown")
            if category not in facts_by_category:
                facts_by_category[category] = []
            facts_by_category[category].append(fact)
        
        # Print facts by category
        for category, category_facts in facts_by_category.items():
            print(f"  {category}: {len(category_facts)}")
            # Print first 5 facts in each category
            for i, fact in enumerate(category_facts[:5]):
                print(f"    {fact['metric_name']}: {fact['value']}")
    
    # Print metrics
    metrics = financials.get("metrics", {})
    print(f"\nMetrics: {len(metrics)}")
    for name, value in list(metrics.items())[:10]:  # Show first 10 metrics
        print(f"  {name}: {value}")
    
    # Print statements
    statements = financials.get("statements", {})
    print(f"\nStatements: {len(statements)}")
    for name, statement in statements.items():
        metadata = statement.get("metadata", {})
        category = metadata.get("category", "unknown")
        print(f"  {name} ({category})")
        
        # Print statement data sample
        data = statement.get("data", [])
        if data:
            print(f"    Rows: {len(data)}")
            if len(data) > 0:
                # Print first row keys
                first_row = data[0]
                print(f"    Columns: {list(first_row.keys())[:5]}...")
    
    # Print tables
    tables = financials.get("tables", {})
    print(f"\nTables: {len(tables)}")
    for name, table in tables.items():
        print(f"  {name}")
        print(f"    Rows: {len(table)}")
    
    # Print relationships
    relationships = financials.get("relationships", [])
    print(f"\nRelationships: {len(relationships)}")
    for i, rel in enumerate(relationships[:5]):  # Show first 5 relationships
        print(f"  {rel.get('source')} -> {rel.get('target')} ({rel.get('type')})")
    
    # Print ratios
    ratios = financials.get("ratios", {})
    print(f"\nRatios: {len(ratios)}")
    for name, value in ratios.items():
        print(f"  {name}: {value:.4f}")
    
    return financials

def compare_extraction_methods(ticker, accession_number):
    """Compare different XBRL extraction methods.
    
    Args:
        ticker: Company ticker symbol
        accession_number: SEC accession number
    """
    # Initialize extractors
    parallel_extractor = ParallelXBRLExtractor(
        cache_dir="data/xbrl_cache_parallel",
        max_workers=1,
        rate_limit=0.2
    )
    
    # Generate a filing ID
    filing_id = f"{ticker}_{accession_number.replace('-', '_')}"
    
    # Extract using parallel extractor
    print("\n=== Testing Parallel XBRL Extractor ===")
    start_time = time.time()
    parallel_results = parallel_extractor.extract_financials(
        ticker=ticker,
        filing_id=filing_id,
        accession_number=accession_number
    )
    parallel_time = time.time() - start_time
    
    # Print comparison
    print("\n=== Extraction Method Comparison ===")
    print(f"Parallel Extractor Time: {parallel_time:.2f} seconds")
    
    # Compare fact counts
    parallel_facts = len(parallel_results.get("facts", []))
    print(f"Parallel Extractor Facts: {parallel_facts}")
    
    # Compare metric counts
    parallel_metrics = len(parallel_results.get("metrics", {}))
    print(f"Parallel Extractor Metrics: {parallel_metrics}")
    
    # Compare statement counts
    parallel_statements = len(parallel_results.get("statements", {}))
    print(f"Parallel Extractor Statements: {parallel_statements}")
    
    # Compare ratio counts
    parallel_ratios = len(parallel_results.get("ratios", {}))
    print(f"Parallel Extractor Ratios: {parallel_ratios}")
    
    return {
        "parallel": parallel_results
    }

def test_batch_extraction(tickers, accession_numbers):
    """Test batch XBRL extraction.
    
    Args:
        tickers: List of company ticker symbols
        accession_numbers: List of SEC accession numbers
    """
    # Initialize the extractor
    extractor = ParallelXBRLExtractor(
        cache_dir="data/xbrl_cache",
        max_workers=4,
        rate_limit=0.2
    )
    
    # Prepare companies data
    companies = []
    for i, ticker in enumerate(tickers):
        if i < len(accession_numbers):
            accession_number = accession_numbers[i]
            filing_id = f"{ticker}_{accession_number.replace('-', '_')}"
            
            company = {
                "ticker": ticker,
                "filings": [
                    {
                        "filing_id": filing_id,
                        "accession_number": accession_number
                    }
                ]
            }
            companies.append(company)
    
    # Start timer
    start_time = time.time()
    
    # Extract financials in parallel
    results = extractor.extract_financials_for_companies(companies)
    
    # End timer
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Print results
    print(f"\n=== Batch XBRL Extraction Results ===")
    print(f"Processed {len(companies)} companies in {elapsed:.2f} seconds")
    
    for ticker, filings in results.items():
        print(f"\nCompany: {ticker}")
        print(f"Filings: {len(filings)}")
        
        for filing in filings:
            if "error" in filing:
                print(f"  Error: {filing['error']}")
            else:
                print(f"  Filing: {filing['accession_number']}")
                print(f"  Facts: {len(filing.get('facts', []))}")
                print(f"  Metrics: {len(filing.get('metrics', {}))}")
                print(f"  Statements: {len(filing.get('statements', {}))}")
                print(f"  Ratios: {len(filing.get('ratios', {}))}")
    
    return results

def store_in_duckdb(financial_data):
    """Store financial data in DuckDB.
    
    Args:
        financial_data: Dictionary mapping tickers to lists of financial data
    """
    # Initialize the store
    store = OptimizedDuckDBStore(db_path="data/improved_financial_data.duckdb")
    
    # Get database stats before
    stats_before = store.get_database_stats()
    
    print("\n=== Database Statistics (Before) ===")
    print(f"Companies: {stats_before.get('company_count', 0)}")
    print(f"Filings: {stats_before.get('filing_count', 0)}")
    print(f"Financial Facts: {stats_before.get('fact_count', 0)}")
    print(f"Time Series Metrics: {stats_before.get('time_series_count', 0)}")
    print(f"Financial Ratios: {stats_before.get('ratio_count', 0)}")
    
    # Flatten financial data for batch storage
    all_financial_data = []
    for ticker, filings in financial_data.items():
        all_financial_data.extend(filings)
    
    # Start timer
    start_time = time.time()
    
    # Store financial data in batch
    stored_count = store.store_financial_data_batch(all_financial_data)
    
    # End timer
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n=== Batch Storage Results ===")
    print(f"Stored {stored_count} financial data records in {elapsed:.2f} seconds")
    
    # Get database stats after
    stats_after = store.get_database_stats()
    
    print("\n=== Database Statistics (After) ===")
    print(f"Companies: {stats_after.get('company_count', 0)}")
    print(f"Filings: {stats_after.get('filing_count', 0)}")
    print(f"Financial Facts: {stats_after.get('fact_count', 0)}")
    print(f"Time Series Metrics: {stats_after.get('time_series_count', 0)}")
    print(f"Financial Ratios: {stats_after.get('ratio_count', 0)}")
    
    return stats_after

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test improved XBRL extraction")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Company ticker symbol")
    parser.add_argument("--accession", type=str, default="0000320193-22-000108", help="SEC accession number")
    parser.add_argument("--mode", type=str, choices=["single", "compare", "batch"], default="single", help="Test mode")
    parser.add_argument("--batch-tickers", type=str, nargs="+", default=["AAPL", "MSFT", "GOOGL"], help="Batch tickers")
    parser.add_argument("--batch-accessions", type=str, nargs="+", default=["0000320193-22-000108", "0000789019-22-000072", "0001652044-22-000071"], help="Batch accession numbers")
    
    args = parser.parse_args()
    
    # Create data directories if they don't exist
    os.makedirs("data/xbrl_cache", exist_ok=True)
    os.makedirs("data/xbrl_cache_parallel", exist_ok=True)
    
    if args.mode == "single":
        # Test single filing extraction
        financials = test_xbrl_extraction(args.ticker, args.accession)
        
        # Save results to file
        output_file = f"data/{args.ticker}_{args.accession.replace('-', '_')}_improved.json"
        with open(output_file, "w") as f:
            json.dump(financials, f, indent=2)
        
        print(f"\nFull data saved to {output_file}")
        
    elif args.mode == "compare":
        # Compare extraction methods
        results = compare_extraction_methods(args.ticker, args.accession)
        
    elif args.mode == "batch":
        # Test batch extraction
        results = test_batch_extraction(args.batch_tickers, args.batch_accessions)
        
        # Store in DuckDB
        stats = store_in_duckdb(results)
