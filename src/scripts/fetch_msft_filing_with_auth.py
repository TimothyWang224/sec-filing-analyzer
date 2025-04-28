"""
Fetch Microsoft's first 2022 filing (0000789019-22-000001) using the edgar package with proper authentication.
"""

import asyncio
import json
import os
from pathlib import Path

import edgar
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


def set_edgar_identity():
    """Set the edgar identity from environment variables or prompt the user."""
    # Check if EDGAR_IDENTITY is set in environment variables
    edgar_identity = os.getenv("EDGAR_IDENTITY")

    if not edgar_identity:
        # Prompt the user for their identity
        print("SEC requires identification for API requests.")
        name = input("Enter your name: ")
        email = input("Enter your email address: ")
        edgar_identity = f"{name} {email}"

        # Suggest adding to .env file
        print("\nConsider adding the following line to a .env file in the project root:")
        print(f'EDGAR_IDENTITY="{edgar_identity}"')

    # Set the identity in the edgar package
    edgar.set_identity(edgar_identity)
    print(f"Set edgar identity to: {edgar_identity}")


async def fetch_specific_filing():
    """
    Fetch a specific Microsoft filing using the edgar package.
    Accession number: 0000789019-22-000001
    """
    print("Fetching Microsoft (MSFT) filing with accession number 0000789019-22-000001...")

    # Create output directory
    output_dir = Path("data/msft_filings")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Get the Microsoft entity
        msft = edgar.get_entity("MSFT")
        print(f"Found Microsoft entity with CIK: {msft.cik}")

        # Get all filings
        filings = msft.get_filings()
        print(f"Retrieved {len(filings)} filings")

        # Find the specific filing by accession number
        target_accession = "0000789019-22-000001"
        found_filing = None

        for filing in filings:
            if filing.accession_number == target_accession:
                found_filing = filing
                break

        if found_filing:
            print(f"Found filing: {found_filing.form} filed on {found_filing.filing_date}")
            print(f"Filing URL: {found_filing.filing_url}")

            # Get the filing details
            filing_text = found_filing.text

            # Save the filing text
            output_file = output_dir / f"MSFT_{target_accession}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(filing_text)
            print(f"Saved filing text to {output_file}")

            # Try to get XBRL data if available
            if hasattr(found_filing, "is_xbrl") and found_filing.is_xbrl:
                print("Filing has XBRL data")
                xbrl_data = found_filing.xbrl

                # Save XBRL data
                xbrl_file = output_dir / f"MSFT_{target_accession}_xbrl.json"
                with open(xbrl_file, "w", encoding="utf-8") as f:
                    json.dump(xbrl_data, f, indent=2, default=str)
                print(f"Saved XBRL data to {xbrl_file}")
            else:
                print("Filing does not have XBRL data")

            # Try to extract document content
            if hasattr(found_filing, "document"):
                document = found_filing.document
                doc_file = output_dir / f"MSFT_{target_accession}_document.html"
                with open(doc_file, "w", encoding="utf-8") as f:
                    f.write(str(document))
                print(f"Saved document content to {doc_file}")

            return found_filing
        else:
            print(f"Filing with accession number {target_accession} not found")

            # Print the first few filings from 2022 to help identify the issue
            print("\nFirst few filings from 2022:")
            count = 0
            for filing in filings:
                filing_date = str(filing.filing_date)
                if "2022" in filing_date:
                    print(f"Accession: {filing.accession_number}, Form: {filing.form}, Date: {filing_date}")
                    count += 1
                    if count >= 5:  # Just show a few
                        break

            return None

    except Exception as e:
        print(f"Error fetching filing: {e}")
        return None


def main():
    """Main function to run the script."""
    try:
        # Set edgar identity
        set_edgar_identity()

        print("Getting Microsoft entity...")
        # Test edgar package functionality
        msft = edgar.get_entity("MSFT")
        print(f"Microsoft CIK: {msft.cik}")

        # Print edgar package version
        print(f"Edgar package version: {edgar.__version__ if hasattr(edgar, '__version__') else 'unknown'}")

        # Run the async function
        print("Running async function to fetch filing...")
        asyncio.run(fetch_specific_filing())
    except Exception as e:
        print(f"Error in main function: {e}")


if __name__ == "__main__":
    main()
