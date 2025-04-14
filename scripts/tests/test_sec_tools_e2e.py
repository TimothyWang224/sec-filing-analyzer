"""
End-to-End Test Script for SEC Tools

This script tests the integration of all SEC tools in the financial environment.
"""

import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse

from src.environments.financial import FinancialEnvironment
from src.tools.sec_semantic_search import SECSemanticSearchTool
from src.tools.sec_graph_query import SECGraphQueryTool
from src.tools.sec_financial_data import SECFinancialDataTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_semantic_search(env: FinancialEnvironment, query: str, companies: Optional[List[str]] = None):
    """Test semantic search through the environment."""
    action = {
        "tool": "sec_semantic_search",
        "args": {
            "query": query,
            "companies": companies,
            "top_k": 3
        }
    }
    
    logger.info(f"Executing semantic search: {query}")
    result = await env.execute_action(action)
    
    print("\n=== Semantic Search Results ===")
    print(f"Query: {query}")
    print(f"Companies: {companies}")
    print(f"Total Results: {result['total_results']}")
    
    for i, res in enumerate(result['results']):
        print(f"\n--- Result {i+1} ---")
        print(f"Score: {res['score']:.4f}")
        print(f"Company: {res['metadata']['company']} ({res['metadata']['ticker']})")
        print(f"Filing: {res['metadata']['filing_type']} ({res['metadata']['filing_date']})")
        print(f"Text: {res['text'][:200]}...")
    
    return result

async def test_graph_query(env: FinancialEnvironment, ticker: str):
    """Test graph query through the environment."""
    action = {
        "tool": "sec_graph_query",
        "args": {
            "query_type": "company_filings",
            "parameters": {
                "ticker": ticker,
                "limit": 5
            }
        }
    }
    
    logger.info(f"Executing graph query for company: {ticker}")
    result = await env.execute_action(action)
    
    print("\n=== Graph Query Results ===")
    print(f"Company: {ticker}")
    print(f"Query Type: company_filings")
    print(f"Total Results: {len(result['results'])}")
    
    for i, res in enumerate(result['results']):
        print(f"\n--- Filing {i+1} ---")
        print(f"Type: {res.get('filing_type', 'N/A')}")
        print(f"Date: {res.get('filing_date', 'N/A')}")
        print(f"Accession Number: {res.get('accession_number', 'N/A')}")
    
    return result

async def test_financial_data(env: FinancialEnvironment, ticker: str):
    """Test financial data query through the environment."""
    action = {
        "tool": "sec_financial_data",
        "args": {
            "query_type": "financial_facts",
            "parameters": {
                "ticker": ticker,
                "metrics": ["Revenue", "NetIncome"],
                "filing_type": "10-K"
            }
        }
    }
    
    logger.info(f"Executing financial data query for company: {ticker}")
    result = await env.execute_action(action)
    
    print("\n=== Financial Data Results ===")
    print(f"Company: {ticker}")
    print(f"Query Type: financial_facts")
    print(f"Total Results: {len(result['results'])}")
    
    for i, res in enumerate(result['results']):
        print(f"\n--- Fact {i+1} ---")
        print(f"Metric: {res.get('metric_name', 'N/A')}")
        print(f"Value: {res.get('value', 'N/A')}")
        print(f"Period End Date: {res.get('period_end_date', 'N/A')}")
        print(f"Filing Type: {res.get('filing_type', 'N/A')}")
    
    return result

async def run_e2e_test(ticker: str, query: str):
    """Run end-to-end test of all tools."""
    # Initialize environment
    env = FinancialEnvironment()
    
    # Test each tool
    try:
        # 1. Semantic Search
        await test_semantic_search(env, query, [ticker])
        
        # 2. Graph Query
        await test_graph_query(env, ticker)
        
        # 3. Financial Data
        await test_financial_data(env, ticker)
        
        print("\n=== End-to-End Test Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Error in end-to-end test: {str(e)}")
        raise

def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test SEC Tools End-to-End")
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Company ticker symbol")
    parser.add_argument("--query", type=str, default="revenue growth and profitability",
                        help="Search query text")
    
    args = parser.parse_args()
    
    # Run the end-to-end test
    asyncio.run(run_e2e_test(
        ticker=args.ticker,
        query=args.query
    ))

if __name__ == "__main__":
    main()
