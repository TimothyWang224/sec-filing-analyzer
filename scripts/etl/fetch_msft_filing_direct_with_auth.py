"""
Fetch Microsoft's first 2022 filing (0000789019-22-000001) directly from SEC EDGAR with proper authentication.
"""

import requests
import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def get_sec_identity():
    """Get the SEC identity from environment variables or prompt the user."""
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
    
    return edgar_identity

def fetch_msft_filing_direct():
    """
    Fetch Microsoft's first 2022 filing (0000789019-22-000001) directly from SEC EDGAR.
    """
    print("Fetching Microsoft filing 0000789019-22-000001 directly from SEC EDGAR...")
    
    # Get SEC identity
    sec_identity = get_sec_identity()
    
    # Create output directory
    output_dir = Path("data/msft_filings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Microsoft's CIK with leading zeros
    cik = "0000789019"
    
    # Target accession number
    accession = "0000789019-22-000001"
    accession_no_dashes = accession.replace("-", "")
    
    # Headers required by SEC
    headers = {
        "User-Agent": sec_identity,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    }
    
    try:
        # First, get the company submissions to find the filing
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        print(f"Fetching submissions from: {submissions_url}")
        
        # Add delay to comply with SEC rate limits
        time.sleep(0.1)
        
        response = requests.get(submissions_url, headers=headers)
        if response.status_code == 200:
            submissions = response.json()
            
            # Save submissions data
            with open(output_dir / "msft_submissions.json", "w") as f:
                json.dump(submissions, f, indent=2)
            print(f"Saved submissions data to {output_dir}/msft_submissions.json")
            
            # Look for our target filing
            found = False
            if "filings" in submissions and "recent" in submissions["filings"]:
                recent = submissions["filings"]["recent"]
                
                if "accessionNumber" in recent:
                    for i, acc in enumerate(recent["accessionNumber"]):
                        if acc == accession:
                            found = True
                            form = recent["form"][i]
                            filing_date = recent["filingDate"][i]
                            print(f"Found filing: {form} filed on {filing_date}")
                            
                            # Now get the actual filing content
                            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{accession}-index.htm"
                            print(f"Filing URL: {filing_url}")
                            
                            # Add delay to comply with SEC rate limits
                            time.sleep(0.1)
                            
                            filing_response = requests.get(filing_url, headers=headers)
                            if filing_response.status_code == 200:
                                # Save the filing index page
                                with open(output_dir / f"MSFT_{accession}_index.html", "w", encoding="utf-8") as f:
                                    f.write(filing_response.text)
                                print(f"Saved filing index to {output_dir}/MSFT_{accession}_index.html")
                                
                                # Try to find the main document link
                                import re
                                doc_links = re.findall(r'href="([^"]+\.htm)"', filing_response.text)
                                if doc_links:
                                    main_doc = doc_links[0]
                                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{main_doc}"
                                    print(f"Main document URL: {doc_url}")
                                    
                                    # Add delay to comply with SEC rate limits
                                    time.sleep(0.1)
                                    
                                    doc_response = requests.get(doc_url, headers=headers)
                                    if doc_response.status_code == 200:
                                        # Save the main document
                                        with open(output_dir / f"MSFT_{accession}_main.html", "w", encoding="utf-8") as f:
                                            f.write(doc_response.text)
                                        print(f"Saved main document to {output_dir}/MSFT_{accession}_main.html")
                                    else:
                                        print(f"Failed to get main document: {doc_response.status_code}")
                                else:
                                    print("Could not find main document link")
                            else:
                                print(f"Failed to get filing index: {filing_response.status_code}")
                            
                            break
            
            if not found:
                print(f"Filing with accession number {accession} not found in submissions")
                
                # Print the first few filings from 2022 to help identify the issue
                print("\nFirst few filings from 2022:")
                if "filings" in submissions and "recent" in submissions["filings"]:
                    recent = submissions["filings"]["recent"]
                    count = 0
                    for i, date in enumerate(recent.get("filingDate", [])):
                        if "2022" in date:
                            acc = recent["accessionNumber"][i]
                            form = recent["form"][i]
                            print(f"Accession: {acc}, Form: {form}, Date: {date}")
                            count += 1
                            if count >= 5:  # Just show a few
                                break
        else:
            print(f"Failed to get submissions: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error fetching filing: {e}")

if __name__ == "__main__":
    fetch_msft_filing_direct()
