# ETL Scripts

This directory contains scripts for extracting, transforming, and loading SEC filing data.

## Master ETL Script

The `master_etl.py` script provides a centralized entry point for the complete ETL process:

1. Retrieving SEC filings using edgar
2. Chunking, embedding, and upserting to a vector store
3. Indexing the vectors in FAISS with optimized parameters
4. Creating a graph over the chunks and filings in Neo4j
5. Extracting XBRL facts and storing them in DuckDB

### Basic Usage

```bash
python scripts/etl/master_etl.py --tickers AAPL MSFT --start-date 2022-01-01 --end-date 2023-01-01
```

### Advanced Options

#### Input Options
- `--tickers AAPL MSFT NVDA`: List of company ticker symbols
- `--tickers-file path/to/tickers.json`: Path to a JSON file containing a list of ticker symbols
- `--filing-types 10-K 10-Q 8-K`: List of filing types to process (default: 10-K, 10-Q)
- `--start-date 2022-01-01`: Start date for filing range (required)
- `--end-date 2023-01-01`: End date for filing range (required)

#### Configuration Options
- `--config-file path/to/config.json`: Path to a JSON configuration file
- `--save-config`: Save the configuration to a file
- `--config-output path/to/output.json`: Path to save the configuration file

#### Neo4j Options
- `--no-neo4j`: Disable Neo4j and use in-memory graph store instead
- `--neo4j-url bolt://localhost:7687`: Neo4j server URL
- `--neo4j-username neo4j`: Neo4j username
- `--neo4j-password password`: Neo4j password
- `--neo4j-database neo4j`: Neo4j database name

#### DuckDB Options
- `--db-path data/financial_data.duckdb`: Path to the DuckDB database file

#### Pipeline Options
- `--no-semantic`: Disable semantic processing (chunking, embedding, etc.)
- `--no-quantitative`: Disable quantitative processing (XBRL extraction, etc.)
- `--no-parallel`: Disable parallel processing
- `--max-workers 4`: Maximum number of worker threads for parallel processing
- `--batch-size 100`: Batch size for embedding generation
- `--rate-limit 0.1`: Minimum time between API requests in seconds

#### FAISS Indexing Options
- `--index-type hnsw`: FAISS index type to use (choices: flat, ivf, hnsw, ivfpq)
- `--use-gpu`: Use GPU acceleration for FAISS if available

##### IVF Parameters
- `--ivf-nlist 100`: Number of clusters for IVF indexes
- `--ivf-nprobe 10`: Number of clusters to visit during search

##### HNSW Parameters
- `--hnsw-m 32`: Number of connections per element
- `--hnsw-ef-construction 400`: Size of dynamic list during construction
- `--hnsw-ef-search 200`: Size of dynamic list during search

#### Additional Options
- `--retry-failed`: Retry failed companies from previous runs
- `--max-retries 3`: Maximum number of retries for failed companies
- `--delay-between-companies 1`: Delay in seconds between processing companies
- `--force-rebuild-index`: Force rebuild of FAISS index even if it exists

### Example: Optimized HNSW Configuration

For optimal performance with HNSW indexing:

```bash
python scripts/etl/master_etl.py \
  --tickers AAPL MSFT NVDA \
  --start-date 2022-01-01 \
  --end-date 2023-01-01 \
  --index-type hnsw \
  --hnsw-m 32 \
  --hnsw-ef-construction 400 \
  --hnsw-ef-search 200 \
  --max-workers 8 \
  --batch-size 50
```

### Example: IVF Configuration

For a balance of speed and accuracy with IVF indexing:

```bash
python scripts/etl/master_etl.py \
  --tickers AAPL MSFT NVDA \
  --start-date 2022-01-01 \
  --end-date 2023-01-01 \
  --index-type ivf \
  --ivf-nlist 100 \
  --ivf-nprobe 10 \
  --max-workers 8 \
  --batch-size 50
```

### Saving and Reusing Configuration

You can save your configuration to a file and reuse it later:

```bash
# Save configuration
python scripts/etl/master_etl.py \
  --tickers AAPL MSFT \
  --start-date 2022-01-01 \
  --end-date 2023-01-01 \
  --index-type hnsw \
  --save-config \
  --config-output data/config/my_etl_config.json

# Reuse configuration
python scripts/etl/master_etl.py \
  --config-file data/config/my_etl_config.json \
  --tickers NVDA  # Override specific parameters
```

## Other ETL Scripts

- `run_nvda_etl.py`: Process SEC filings for NVIDIA Corporation (NVDA)
- `run_multi_company_etl.py`: Process SEC filings for multiple companies
- `run_multi_company_etl_parallel.py`: Process SEC filings for multiple companies in parallel
- `update_etl_pipeline.py`: Update the ETL pipeline
- `extract_xbrl_data.py`: Extract XBRL data from SEC filings
- `extract_xbrl_direct.py`: Extract XBRL data directly from SEC filings
- `fetch_msft_filing.py`: Fetch Microsoft SEC filings
- `list_msft_filings.py`: List Microsoft SEC filings
- `reprocess_aapl_filing.py`: Reprocess Apple SEC filings
- `reprocess_zero_vector_filings.py`: Reprocess filings with zero vectors
