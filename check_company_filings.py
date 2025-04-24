import duckdb
import pandas as pd

# Connect to the database
try:
    conn = duckdb.connect('data/db_backup/improved_financial_data.duckdb', read_only=True)
    print("Successfully connected to improved_financial_data.duckdb")
except Exception as e:
    print(f"Error connecting to improved_financial_data.duckdb: {e}")
    print("Trying to connect to financial_data.duckdb instead...")
    try:
        conn = duckdb.connect('data/db_backup/financial_data.duckdb', read_only=True)
        print("Successfully connected to financial_data.duckdb")
    except Exception as e:
        print(f"Error connecting to financial_data.duckdb: {e}")
        exit(1)

# Check tables
print("\nTables in database:")
tables = conn.execute('SHOW TABLES').fetchdf()
print(tables)

# First, let's examine the schema of the tables to understand their structure
print("\nSchema of filings table:")
try:
    filings_schema = conn.execute('DESCRIBE filings').fetchdf()
    print(filings_schema)
except Exception as e:
    print(f"Error getting filings schema: {e}")

print("\nSchema of time_series_view table:")
try:
    time_series_schema = conn.execute('DESCRIBE time_series_view').fetchdf()
    print(time_series_schema)
except Exception as e:
    print(f"Error getting time_series_view schema: {e}")

# Check companies
print("\nCompanies in database:")
try:
    companies = conn.execute('SELECT * FROM companies').fetchdf()
    print(companies)
except Exception as e:
    print(f"Error querying companies: {e}")

# Check if company_id exists in filings table
company_id_exists = 'company_id' in filings_schema['column_name'].values if 'filings_schema' in locals() else False
ticker_exists = 'ticker' in filings_schema['column_name'].values if 'filings_schema' in locals() else False

# Check filings for each company
tickers = ['NVDA', 'GOOGL', 'AAPL', 'MSFT']
for ticker in tickers:
    print(f"\nFilings for {ticker}:")
    try:
        # Adjust query based on schema
        if company_id_exists:
            # First get company_id for the ticker
            company_id_query = f"SELECT company_id FROM companies WHERE ticker = '{ticker}'"
            company_id_result = conn.execute(company_id_query).fetchone()

            if company_id_result:
                company_id = company_id_result[0]
                filings_query = f"SELECT * FROM filings WHERE company_id = {company_id} ORDER BY filing_date DESC"
            else:
                print(f"No company_id found for ticker {ticker}")
                continue
        elif ticker_exists:
            filings_query = f"SELECT * FROM filings WHERE ticker = '{ticker}' ORDER BY filing_date DESC"
        else:
            # Try to find any column that might contain the ticker
            print(f"Searching for {ticker} in filings table...")
            sample_query = f"SELECT * FROM filings LIMIT 5"
            sample = conn.execute(sample_query).fetchdf()
            print(f"Sample filings data: {sample}")
            continue

        filings = conn.execute(filings_query).fetchdf()

        if len(filings) > 0:
            # Group by fiscal year to see how many filings per year
            if 'fiscal_year' in filings.columns:
                filings_by_year = filings.groupby('fiscal_year').size().reset_index(name='count')
                print(f"Filings by year for {ticker}:")
                print(filings_by_year)

            # Show the most recent filings
            print(f"Most recent filings for {ticker} (up to 5):")
            columns_to_show = [col for col in ['id', 'filing_id', 'company_id', 'ticker', 'filing_type', 'filing_date', 'fiscal_year', 'fiscal_quarter'] if col in filings.columns]
            print(filings.head(5)[columns_to_show])

            # Count total filings
            print(f"Total filings for {ticker}: {len(filings)}")
        else:
            print(f"No filings found for {ticker}")
    except Exception as e:
        print(f"Error querying filings for {ticker}: {e}")

# Check time series metrics for each company
for ticker in tickers:
    print(f"\nTime series metrics for {ticker}:")
    try:
        # Try time_series_view first
        metrics_query = f"SELECT DISTINCT metric_name FROM time_series_view WHERE ticker = '{ticker}'"
        metrics = conn.execute(metrics_query).fetchdf()

        if len(metrics) > 0:
            print(f"Available metrics for {ticker}:")
            print(metrics)

            # Check revenue data for the last 3 years
            print(f"Revenue data for {ticker} (last 3 years):")
            revenue_query = f"""
                SELECT fiscal_year, fiscal_period, value
                FROM time_series_view
                WHERE ticker = '{ticker}' AND metric_name = 'Revenue'
                ORDER BY fiscal_year DESC, fiscal_period DESC
                LIMIT 12
            """
            revenue = conn.execute(revenue_query).fetchdf()
            print(revenue)
        else:
            print(f"No metrics found for {ticker} in time_series_view")

            # Try metrics table as fallback
            metrics_query = f"SELECT DISTINCT name FROM metrics WHERE ticker = '{ticker}'"
            metrics = conn.execute(metrics_query).fetchdf()

            if len(metrics) > 0:
                print(f"Available metrics for {ticker} in metrics table:")
                print(metrics)
            else:
                print(f"No metrics found for {ticker} in metrics table")
    except Exception as e:
        print(f"Error querying metrics for {ticker}: {e}")

# Close the connection
conn.close()
