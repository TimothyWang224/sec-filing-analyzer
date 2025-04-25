from datetime import datetime, timedelta

from src.sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline

# Set up date range for 3 years
today = datetime.now()
three_years_ago = today - timedelta(days=365 * 3)
start_date = three_years_ago.strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")

print(f"Testing ETL pipeline with date range: {start_date} to {end_date}")

# Initialize pipeline
pipeline = SECFilingETLPipeline(
    process_semantic=False,  # Set to False to avoid embedding issues
    process_quantitative=False,  # Set to False to speed up the test
)

# Process NVDA filings with limit=10
result = pipeline.process_company_filings(
    ticker="NVDA",
    filing_types=["10-K", "10-Q", "8-K"],
    start_date=start_date,
    end_date=end_date,
    limit=10,  # Limit to 10 filings for testing
)

# Print result
print(f"Result: {result}")
if "filings_processed" in result:
    print(f"Filings processed: {result['filings_processed']}")
if "error" in result:
    print(f"Error: {result['error']}")
