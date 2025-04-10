import edgar
import asyncio
import json
from pathlib import Path

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
            if hasattr(found_filing, 'is_xbrl') and found_filing.is_xbrl:
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
            if hasattr(found_filing, 'document'):
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
            for filing in filings:
                if "2022" in str(filing.filing_date):
                    print(f"Accession: {filing.accession_number}, Form: {filing.form}, Date: {filing.filing_date}")
                    if len(filing.accession_number) > 20:  # Just show a few
                        break
            
            return None
    
    except Exception as e:
        print(f"Error fetching filing: {e}")
        return None

def main():
    """Main function to run the script."""
    asyncio.run(fetch_specific_filing())

if __name__ == "__main__":
    main()
