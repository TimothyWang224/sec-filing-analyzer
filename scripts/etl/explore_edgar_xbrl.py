"""
Explore Edgar XBRL Capabilities

This script explores the XBRL extraction capabilities of the edgar package.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import edgar
import edgar


def explore_entity_capabilities():
    """Explore the capabilities of the Entity class."""
    logger.info("Exploring Entity capabilities...")

    # Get an entity
    entity = edgar.get_entity("AAPL")

    # Print entity information
    print(f"Entity: {entity.name} (CIK: {entity.cik})")
    print(f"Entity attributes: {dir(entity)}")

    # Get filings
    filings = entity.get_filings()

    # Print filing information
    print(f"\nFilings: {len(filings)}")
    for i, filing in enumerate(list(filings)[:3]):
        print(f"\nFiling {i + 1}:")
        print(f"  Attributes: {dir(filing)}")
        print(f"  Accession Number: {filing.accession_number}")
        print(f"  Form: {filing.form}")
        print(f"  Filing Date: {filing.filing_date}")

        # Try to get XBRL data
        try:
            print("\nTrying to get XBRL data...")
            # Check if there's a method to get XBRL data directly
            if hasattr(filing, "get_xbrl_data"):
                xbrl_data = filing.get_xbrl_data()
                print(f"  XBRL data: {xbrl_data}")
            else:
                print("  No direct method to get XBRL data")
        except Exception as e:
            print(f"  Error getting XBRL data: {e}")


def explore_xbrl_capabilities():
    """Explore the capabilities of the XBRL module."""
    logger.info("Exploring XBRL capabilities...")

    # Print XBRL module information
    print(f"XBRL module attributes: {dir(edgar.xbrl)}")

    # Check XBRLData class
    print(f"\nXBRLData class attributes: {dir(edgar.xbrl.XBRLData)}")

    # Try to get XBRL data for a filing
    try:
        print("\nTrying to get XBRL data for a filing...")

        # Get an entity
        entity = edgar.get_entity("AAPL")

        # Get filings
        filings = entity.get_filings()
        # Filter for 10-K filings
        k_filings = [f for f in filings if f.form == "10-K"]
        if k_filings:
            filing = k_filings[0]
            print(f"Filing: {filing.accession_number} ({filing.form})")

            # Construct the filing URL
            cik = entity.cik
            accession_number = filing.accession_number
            accession_clean = accession_number.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{accession_number}-index.htm"
            print(f"Filing URL: {filing_url}")

            # Try to get XBRL data
            try:
                xbrl_data = edgar.xbrl.XBRLData.from_filing(filing_url)
                print(f"XBRL data: {xbrl_data}")
                print(f"XBRL data attributes: {dir(xbrl_data)}")

                # Check if there are statements
                if hasattr(xbrl_data, "statements"):
                    print(f"\nStatements: {list(xbrl_data.statements.keys())}")

                    # Print the first statement
                    if xbrl_data.statements:
                        statement_name = list(xbrl_data.statements.keys())[0]
                        statement = xbrl_data.statements[statement_name]
                        print(f"\nStatement: {statement_name}")
                        print(f"Statement attributes: {dir(statement)}")

                        # Try to convert to pandas
                        if hasattr(statement, "to_pandas"):
                            df = statement.to_pandas()
                            print(f"\nStatement as DataFrame:")
                            print(df.head())

                # Check if there are facts
                if hasattr(xbrl_data, "facts"):
                    print(f"\nFacts: {len(xbrl_data.facts)}")

                    # Print some facts
                    for i, fact in enumerate(list(xbrl_data.facts.values())[:5]):
                        print(f"\nFact {i + 1}:")
                        print(f"  Attributes: {dir(fact)}")
                        print(f"  Name: {fact.name if hasattr(fact, 'name') else 'N/A'}")
                        print(f"  Value: {fact.value if hasattr(fact, 'value') else 'N/A'}")
                        print(f"  Context: {fact.context if hasattr(fact, 'context') else 'N/A'}")

                # Check if there's a method to get all US GAAP facts
                if hasattr(xbrl_data, "get_us_gaap_facts"):
                    us_gaap_facts = xbrl_data.get_us_gaap_facts()
                    print(f"\nUS GAAP Facts: {len(us_gaap_facts)}")
                else:
                    print("\nNo direct method to get US GAAP facts")

                    # Try to filter facts by namespace
                    us_gaap_facts = {}
                    for name, fact in xbrl_data.facts.items():
                        if name.startswith("us-gaap:"):
                            us_gaap_facts[name] = fact

                    print(f"\nFiltered US GAAP Facts: {len(us_gaap_facts)}")

                    # Print some US GAAP facts
                    for i, (name, fact) in enumerate(list(us_gaap_facts.items())[:5]):
                        print(f"\nUS GAAP Fact {i + 1}:")
                        print(f"  Name: {name}")
                        print(f"  Value: {fact.value if hasattr(fact, 'value') else 'N/A'}")

            except Exception as e:
                print(f"Error getting XBRL data: {e}")
        else:
            print("No filings found")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("\n=== Exploring Entity Capabilities ===\n")
    explore_entity_capabilities()

    print("\n=== Exploring XBRL Capabilities ===\n")
    explore_xbrl_capabilities()
