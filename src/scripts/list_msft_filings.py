"""
List Microsoft's filings using the edgar package with proper authentication.
"""

import os
from datetime import datetime

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


def list_msft_filings():
    """List Microsoft's filings using the edgar package."""
    try:
        # Set edgar identity
        set_edgar_identity()

        print("Getting Microsoft entity...")
        # Get Microsoft entity
        msft = edgar.get_entity("MSFT")
        print(f"Microsoft CIK: {msft.cik}")

        # Get filings
        print("Getting filings...")
        filings = msft.get_filings()
        print(f"Retrieved {len(filings)} filings")

        # Print filings from 2022
        print("\nFilings from 2022:")
        count_2022 = 0

        for filing in filings:
            filing_date = filing.filing_date

            # Check if filing date is a datetime object
            if isinstance(filing_date, datetime):
                filing_date_str = filing_date.strftime("%Y-%m-%d")
            else:
                filing_date_str = str(filing_date)

            if "2022" in filing_date_str:
                print(f"Accession: {filing.accession_number}, Form: {filing.form}, Date: {filing_date_str}")
                count_2022 += 1

        print(f"\nTotal 2022 filings: {count_2022}")

        # Check for specific accession number
        target_accession = "0000789019-22-000001"
        print(f"\nLooking for filing with accession number: {target_accession}")

        found = False
        for filing in filings:
            if filing.accession_number == target_accession:
                found = True
                print(f"Found filing: {filing.form} filed on {filing.filing_date}")
                print(f"Filing URL: {filing.filing_url}")
                break

        if not found:
            print(f"Filing with accession number {target_accession} not found")

            # Print the first filing of 2022
            print("\nFirst filing of 2022:")
            for filing in filings:
                filing_date = filing.filing_date

                # Check if filing date is a datetime object
                if isinstance(filing_date, datetime):
                    filing_date_str = filing_date.strftime("%Y-%m-%d")
                else:
                    filing_date_str = str(filing_date)

                if "2022" in filing_date_str:
                    print(f"Accession: {filing.accession_number}, Form: {filing.form}, Date: {filing_date_str}")
                    break

    except Exception as e:
        print(f"Error listing filings: {e}")


if __name__ == "__main__":
    list_msft_filings()
