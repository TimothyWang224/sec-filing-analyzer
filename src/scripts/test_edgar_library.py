"""
Test the edgar library.
"""

import logging

from edgar import Company, set_identity

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Set edgar identity
    print("Setting edgar identity...")
    set_identity("timothy.yi.wang@gmail.com")

    # Test with a company
    ticker = "MSFT"
    print(f"Getting company info for {ticker}...")
    company = Company(ticker)
    print(f"Company CIK: {company.cik}")

    # Get filings for the company
    print(f"Getting filings for {ticker}...")
    filings = company.get_filings(filing_type="10-K", count=1)

    if filings:
        print(f"Found {len(filings)} filings")
        filing = filings[0]
        print(f"Filing accession number: {filing.accession_number}")
        print(f"Filing date: {filing.filing_date}")
        print(f"Filing form: {filing.form}")

        # Download the filing
        print("Downloading filing...")
        filing.download()

        # Check if the filing has XBRL data
        has_xbrl = hasattr(filing, "is_xbrl") and filing.is_xbrl
        print(f"Has XBRL: {has_xbrl}")

        if has_xbrl:
            # Extract XBRL data
            print("Extracting XBRL data...")
            xbrl_data = filing.xbrl()

            # Print some XBRL data
            if hasattr(xbrl_data, "instance"):
                facts = xbrl_data.instance.query_facts(schema="us-gaap")
                print(f"Found {len(facts)} US-GAAP facts")
                if not facts.empty:
                    print("Sample facts:")
                    print(facts.head(5))
    else:
        print("No filings found")


if __name__ == "__main__":
    main()
