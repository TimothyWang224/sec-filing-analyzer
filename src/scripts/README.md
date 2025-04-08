# SEC Filing ETL Scripts

This directory contains scripts for running the SEC Filing ETL pipeline.

## Available Scripts

### `run_nvda_etl.py`

Process SEC filings for NVIDIA Corporation (NVDA).

```bash
python -m scripts.run_nvda_etl NVDA --start-date 2023-01-01 --end-date 2023-12-31 --filing-types 10-K 10-Q
```

### `run_multi_company_etl.py`

Process SEC filings for multiple companies.

#### Usage with direct ticker list:

```bash
python -m scripts.run_multi_company_etl --tickers AAPL MSFT NVDA --start-date 2023-01-01 --end-date 2023-12-31 --filing-types 10-K 10-Q
```

#### Usage with ticker file:

```bash
python -m scripts.run_multi_company_etl --tickers-file data/sample_tickers.json --start-date 2023-01-01 --end-date 2023-12-31 --filing-types 10-K 10-Q
```

#### Options:

- `--tickers`: List of company ticker symbols (e.g., AAPL MSFT NVDA)
- `--tickers-file`: Path to a JSON file containing a list of ticker symbols
- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD)
- `--filing-types`: List of filing types to process (default: 10-K 10-Q)
- `--no-neo4j`: Disable Neo4j and use in-memory graph store instead
- `--neo4j-url`: Neo4j server URL (default: bolt://localhost:7687)
- `--neo4j-username`: Neo4j username (default: neo4j)
- `--neo4j-password`: Neo4j password (default: password)
- `--neo4j-database`: Neo4j database name (default: neo4j)
- `--retry-failed`: Retry failed companies from previous runs
- `--max-retries`: Maximum number of retries for failed companies (default: 3)
- `--delay-between-companies`: Delay in seconds between processing companies (default: 1)

#### Example JSON Ticker Files:

Simple list format:
```json
[
  "AAPL",
  "MSFT",
  "NVDA",
  "GOOGL",
  "AMZN"
]
```

Structured format with metadata:
```json
{
  "tickers": [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN"
  ],
  "metadata": {
    "sector": "Technology",
    "description": "Top 5 technology companies by market cap"
  }
}
```

## Progress Tracking

The `run_multi_company_etl.py` script tracks progress and saves it to JSON files in the `data/etl_progress/` directory. If a run fails, you can use the `--retry-failed` flag to retry only the companies that failed in the previous run.

## Environment Variables

You can configure Neo4j connection details using environment variables:

- `NEO4J_URL`: Neo4j server URL (default: bolt://localhost:7687)
- `NEO4J_USERNAME`: Neo4j username (default: neo4j)
- `NEO4J_PASSWORD`: Neo4j password (default: password)
- `NEO4J_DATABASE`: Neo4j database name (default: neo4j)
