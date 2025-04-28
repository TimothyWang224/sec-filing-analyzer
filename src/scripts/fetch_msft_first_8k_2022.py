"""
Fetch Microsoft's first 8-K filing of 2022 using the edgar package with proper authentication.
"""

import json
import os
from datetime import datetime
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


def fetch_msft_first_8k_2022():
    """Fetch Microsoft's first 8-K filing of 2022."""
    try:
        # Set edgar identity
        set_edgar_identity()

        # Create output directory
        output_dir = Path("data/msft_filings")
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Getting Microsoft entity...")
        # Get Microsoft entity
        msft = edgar.get_entity("MSFT")
        print(f"Microsoft CIK: {msft.cik}")

        # Get filings
        print("Getting filings...")
        filings = msft.get_filings()
        print(f"Retrieved {len(filings)} filings")

        # Find the first 8-K filing of 2022
        print("\nLooking for the first 8-K filing of 2022...")
        first_8k_2022 = None

        for filing in filings:
            filing_date = filing.filing_date

            # Check if filing date is a datetime object
            if isinstance(filing_date, datetime):
                filing_date_str = filing_date.strftime("%Y-%m-%d")
            else:
                filing_date_str = str(filing_date)

            if "2022" in filing_date_str and filing.form == "8-K":
                # Sort by date to find the earliest
                if first_8k_2022 is None or filing_date < first_8k_2022.filing_date:
                    first_8k_2022 = filing

        if first_8k_2022:
            print("Found first 8-K filing of 2022:")
            print(f"Accession: {first_8k_2022.accession_number}")
            print(f"Form: {first_8k_2022.form}")
            print(f"Date: {first_8k_2022.filing_date}")
            print(f"Filing URL: {first_8k_2022.filing_url}")

            # Get the filing details
            print("\nGetting filing details...")
            filing_text = first_8k_2022.text

            # Save the filing text
            output_file = output_dir / "MSFT_first_8k_2022.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(filing_text)
            print(f"Saved filing text to {output_file}")

            # Try to get XBRL data if available
            if hasattr(first_8k_2022, "is_xbrl") and first_8k_2022.is_xbrl:
                print("Filing has XBRL data")
                xbrl_data = first_8k_2022.xbrl

                # Save XBRL data
                xbrl_file = output_dir / "MSFT_first_8k_2022_xbrl.json"
                with open(xbrl_file, "w", encoding="utf-8") as f:
                    json.dump(xbrl_data, f, indent=2, default=str)
                print(f"Saved XBRL data to {xbrl_file}")
            else:
                print("Filing does not have XBRL data")

            # Try to extract document content
            if hasattr(first_8k_2022, "document"):
                document = first_8k_2022.document
                doc_file = output_dir / "MSFT_first_8k_2022_document.html"
                with open(doc_file, "w", encoding="utf-8") as f:
                    f.write(str(document))
                print(f"Saved document content to {doc_file}")

            return first_8k_2022
        else:
            print("No 8-K filings found for 2022")
            return None

    except Exception as e:
        print(f"Error fetching filing: {e}")
        return None


if __name__ == "__main__":
    fetch_msft_first_8k_2022()
