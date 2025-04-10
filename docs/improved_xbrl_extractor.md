# Improved XBRL Extractor

This document explains how to use the improved XBRL extractor to extract financial data from SEC filings and store it in a DuckDB database using the improved schema.

## Overview

The improved XBRL extractor (`ImprovedEdgarXBRLExtractor`) is designed to work with the improved DuckDB schema. It extracts financial data from SEC filings using the edgar library's XBRL parsing capabilities and stores it in a DuckDB database.

Key improvements over the previous extractor:
- Works with the improved DuckDB schema
- Better handling of metrics and facts
- More robust error handling
- Improved logging
- Better documentation

## Prerequisites

Before using the improved XBRL extractor, you need to:

1. Migrate your existing DuckDB database to the new schema (if you have one)
2. Set up the edgar library with proper authentication

### Migrating to the New Schema

To migrate your existing DuckDB database to the new schema, use the migration script:

```bash
python src/scripts/migrate_duckdb_schema.py --old-db data/financial_data.duckdb --new-db data/financial_data_new.duckdb
```

### Setting Up Edgar Authentication

The edgar library requires authentication to access SEC data. Set up your authentication credentials in a `.env` file:

```
EDGAR_IDENTITY=your_email@example.com
```

Or set it programmatically:

```python
from edgar import set_identity
set_identity("your_email@example.com")
```

## Usage

### Extracting Data from a Single Filing

```python
from sec_filing_analyzer.data_processing.improved_edgar_xbrl_extractor import ImprovedEdgarXBRLExtractor

# Create the extractor
extractor = ImprovedEdgarXBRLExtractor(db_path="data/financial_data_new.duckdb")

# Process a single filing
result = extractor.process_filing(ticker="MSFT", accession_number="0000789019-22-000010")

# Close the extractor when done
extractor.close()
```

### Extracting Data from Multiple Filings

```python
from sec_filing_analyzer.data_processing.improved_edgar_xbrl_extractor import ImprovedEdgarXBRLExtractor

# Create the extractor
extractor = ImprovedEdgarXBRLExtractor(db_path="data/financial_data_new.duckdb")

# Process all 10-K and 10-Q filings for a company
result = extractor.process_company(
    ticker="MSFT",
    filing_types=["10-K", "10-Q"],
    limit=10  # Optional: limit the number of filings to process
)

# Close the extractor when done
extractor.close()
```

### Using the Test Script

You can also use the test script to extract data:

```bash
# Process a single filing
python src/scripts/test_improved_xbrl_extractor.py --ticker MSFT --accession 0000789019-22-000010

# Process all 10-K and 10-Q filings for a company
python src/scripts/test_improved_xbrl_extractor.py --ticker MSFT --filing-type 10-K --filing-type 10-Q --limit 10
```

## Data Flow

The improved XBRL extractor follows this data flow:

1. **Download Filing**: Downloads the SEC filing using the edgar library
2. **Store Company**: Stores company information in the database
3. **Store Filing**: Stores filing metadata in the database
4. **Extract XBRL Data**: Extracts XBRL data from the filing
5. **Process Financial Statements**: Processes financial statements (balance sheet, income statement, cash flow)
6. **Process US-GAAP Facts**: Processes US-GAAP facts from the XBRL data
7. **Store Metrics**: Stores metric definitions in the database
8. **Store Facts**: Stores financial facts in the database

## Database Schema

The improved XBRL extractor works with the improved DuckDB schema, which includes:

- **companies**: Stores company information
- **filings**: Stores filing metadata
- **metrics**: Stores metric definitions
- **facts**: Stores financial facts
- **xbrl_tag_mappings**: Maps XBRL tags to standardized metrics

For more information about the improved schema, see [Improved DuckDB Schema](improved_duckdb_schema.md).

## Comparing Old and New Schemas

To compare the old and new schemas, use the comparison script:

```bash
# Compare database statistics
python src/scripts/compare_duckdb_schemas.py --stats

# Compare companies
python src/scripts/compare_duckdb_schemas.py --companies

# Compare filings for a company
python src/scripts/compare_duckdb_schemas.py --filings MSFT

# Compare facts for a filing
python src/scripts/compare_duckdb_schemas.py --facts MSFT 0000789019-22-000010

# Compare time series data
python src/scripts/compare_duckdb_schemas.py --time-series MSFT revenue
```

## Troubleshooting

### Common Issues

#### Filing Not Found

If you get a "Filing not found" error, check that:
- The accession number is correct
- You have set up edgar authentication correctly
- You have internet access

#### Database Connection Error

If you get a database connection error, check that:
- The database file exists
- You have write permissions to the database file
- The database file is not corrupted

#### XBRL Data Not Available

If you get a "Filing processed but no XBRL data available" message, it means the filing does not have XBRL data. This is normal for older filings or certain filing types.

### Logging

The improved XBRL extractor uses Python's logging module to log information, warnings, and errors. By default, logs are written to the console. You can configure logging to write to a file:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="xbrl_extractor.log"
)
```

## Next Steps

After extracting data using the improved XBRL extractor, you can:

1. **Query the Data**: Use the `ImprovedDuckDBStore` class to query the data
2. **Visualize the Data**: Use the Streamlit app to visualize the data
3. **Analyze the Data**: Use the data for financial analysis

For more information about querying the data, see [Improved DuckDB Schema](improved_duckdb_schema.md).
