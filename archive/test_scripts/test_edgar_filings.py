import edgar
from datetime import datetime, timedelta
import sys

# Set up date range for 3 years
today = datetime.now()
three_years_ago = today - timedelta(days=365*3)
date_range = f"{three_years_ago.strftime('%Y-%m-%d')}:{today.strftime('%Y-%m-%d')}"

print(f"Testing edgar library with date range: {date_range}")

# Get company
try:
    company = edgar.Company("NVDA")
    print(f"Successfully retrieved company: {company}")
except Exception as e:
    print(f"Error retrieving company: {e}")
    sys.exit(1)

# Get all filings
try:
    all_filings = company.get_filings(date=date_range)
    print(f"Found {len(all_filings)} total filings for NVDA in the last 3 years")
except Exception as e:
    print(f"Error retrieving all filings: {e}")

# Get 10-K filings
try:
    tenk_filings = company.get_filings(form="10-K", date=date_range)
    print(f"Found {len(tenk_filings)} 10-K filings for NVDA in the last 3 years")
except Exception as e:
    print(f"Error retrieving 10-K filings: {e}")

# Get 10-Q filings
try:
    tenq_filings = company.get_filings(form="10-Q", date=date_range)
    print(f"Found {len(tenq_filings)} 10-Q filings for NVDA in the last 3 years")
except Exception as e:
    print(f"Error retrieving 10-Q filings: {e}")

# Get 8-K filings
try:
    eightk_filings = company.get_filings(form="8-K", date=date_range)
    print(f"Found {len(eightk_filings)} 8-K filings for NVDA in the last 3 years")
except Exception as e:
    print(f"Error retrieving 8-K filings: {e}")

# Print the first few filings to see what they look like
if all_filings:
    print("\nFirst few filings:")
    for i, filing in enumerate(all_filings[:5]):
        print(f"{i+1}. {filing.form} - {filing.filing_date} - {filing.accession_number}")
