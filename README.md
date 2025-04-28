# SEC Filing Analyzer

A comprehensive tool for analyzing SEC filings using graph databases and vector embeddings.

## Features

### Semantic Data Processing

- **Intelligent Chunking**: Semantic chunking of documents with a 1500 token size for optimal retrieval
- **Vector Embeddings**: Generate and store embeddings using OpenAI's API
- **Vector Search**: Efficient similarity search using LlamaIndex
- **Graph Database**: Store filing data in Neo4j with rich relationships
- **Entity Extraction**: Identify companies, people, and financial concepts
- **Topic Analysis**: Extract topics from filing sections

### Quantitative Data Processing

- **XBRL Extraction**: Extract structured financial data from XBRL filings
- **DuckDB Storage**: Store financial data in DuckDB for efficient querying
- **Financial Metrics**: Extract key financial metrics from SEC filings
- **Time-Series Analysis**: Support for time-series analysis of financial data

### Unified ETL Pipeline

- **Modular Architecture**: Process semantic and quantitative data separately or together
- **Parallel Processing**: Process multiple filings in parallel for improved performance
- **Incremental Updates**: Support for incremental updates to the database

## Architecture

The system uses a modular architecture with separate pipelines for semantic and quantitative data processing, as well as an agent-based architecture for analysis:

### Semantic Data Processing

1. **Neo4j Graph Database**:
   - Stores structured filing data
   - Maintains relationships between filings, sections, and entities
   - References vector IDs for semantic search

2. **LlamaIndex Vector Store**:
   - Stores vector embeddings for semantic search
   - Provides efficient similarity search with cosine similarity
   - Maintains metadata for filtering and retrieval

### Quantitative Data Processing

1. **DuckDB Database**:
   - Stores structured financial data extracted from XBRL
   - Provides efficient SQL queries for financial analysis
   - Supports time-series analysis of financial metrics

### Unified ETL Pipeline

The system provides a unified ETL pipeline that can process both semantic and quantitative data, or either one separately, depending on the user's needs.

### Agent Architecture

The system uses an agent-based architecture for analysis:

1. **Phase-Based Execution**:
   - **Planning phase**: Understands tasks and creates structured plans (controlled by `max_planning_iterations`)
   - **Execution phase**: Gathers data and generates initial answers through tool calls (controlled by `max_execution_iterations`)
   - **Refinement phase**: Improves answer quality and presentation (controlled by `max_refinement_iterations`)
   - Overall execution limited by `max_iterations` across all phases

2. **Tool Integration**:
   - Agents can use various tools to gather and analyze data
   - Tool calls are tracked in a Tool Ledger for reference
   - Single tool call approach reduces parameter confusion
   - Robust error handling with adaptive retry strategies
   - Circuit breaker pattern prevents cascading failures
   - Plan-Step ↔ Tool Contract ensures consistent tool execution and memory storage

3. **Specialized Agents**:
   - QA Specialist for answering questions
   - Financial Analyst for financial analysis
   - Risk Analyst for risk assessment
   - Coordinator for orchestrating complex tasks

4. **Plan-Step ↔ Tool Contract**:
   - Formal contract between planning steps and tools using Pydantic models
   - Standardized parameter validation with clear error messages
   - Consistent memory storage with output keys
   - Success criteria for skipping redundant steps
   - Dependency tracking between steps

For more details, see the [Agent Parameters and Phases](docs/agent_parameters.md) documentation.

## Setup

1. Install dependencies:
   ```bash
   # Install using poetry
   poetry install

   # Or install dependencies directly
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`:
   ```
   EDGAR_IDENTITY=your_edgar_identity
   OPENAI_API_KEY=your_openai_api_key
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

3. Initialize Neo4j database:
   ```bash
   python -m sec_filing_analyzer.init_db
   ```

## Running the Application

### Quick Demo

For a quick demonstration of the system's capabilities:

```bash
# Clone and install
git clone https://github.com/TimothyWang224/sec-filing-analyzer.git
cd sec-filing-analyzer
poetry install --no-root

# Run ETL for NVIDIA (downloads real SEC filings)
export OPENAI_API_KEY=sk-...
poetry run python scripts/demo/run_nvda_etl.py --ticker NVDA --years 2023

# For offline testing, use synthetic data
# TEST_MODE=True poetry run python scripts/demo/run_nvda_etl.py --ticker NVDA --years 2023

# Query revenue data
poetry run python scripts/demo/query_revenue.py --ticker NVDA --year 2023
```

The demo uses real NVIDIA SEC filings for authenticity and credibility, with a synthetic data fallback for testing. For more details, see [Demo README](scripts/demo/README.md).

### Full Application

To run the Streamlit application:

#### On Windows

Simply run the batch file in the project root:

```
run_app.bat
```

#### On macOS/Linux

Run the Python launcher script:

```
python run_app.py
```

For more detailed instructions, see [RUNNING.md](RUNNING.md).

## Project Organization

The project is organized as follows:

- **Root Directory**: Contains only essential files like README.md, run_app.py, and configuration files
- **src/**: Contains the main source code for the application
- **scripts/**: Contains utility scripts and tools
  - **scripts/demo/**: Demo scripts for showcasing the project's capabilities
  - **scripts/utils/**: Utility scripts for checking data, monitoring logs, etc.
  - **scripts/data/**: Scripts for data management and manipulation
  - **scripts/maintenance/**: Scripts for project maintenance
  - **scripts/etl/**: Scripts for running the ETL pipeline
  - **scripts/analysis/**: Scripts for analyzing data
  - **scripts/visualization/**: Scripts for visualizing data
- **tests/**: Contains unit and integration tests
- **data/**: Contains data files and databases
  - **data/db_backup/**: Database backup files
  - **data/filings/**: SEC filing data
  - **data/vector_store/**: Vector embeddings and metadata
  - **data/logs/**: Log files
- **docs/**: Contains documentation
- **archive/**: Contains archived files that are no longer needed but kept for reference

For a detailed overview of the directory structure, see [DIRECTORY_STRUCTURE.md](docs/DIRECTORY_STRUCTURE.md).

## Usage

### Unified Pipeline

```python
from sec_filing_analyzer.pipeline import SECFilingETLPipeline

# Initialize pipeline with both semantic and quantitative processing
pipeline = SECFilingETLPipeline(process_semantic=True, process_quantitative=True)

# Process filings for a company
pipeline.process_company(
    ticker="AAPL",
    filing_types=["10-K", "10-Q"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### Semantic Pipeline Only

```python
from sec_filing_analyzer.pipeline import SemanticETLPipeline

# Initialize semantic pipeline
pipeline = SemanticETLPipeline()

# Process a filing
result = pipeline.process_filing(
    ticker="AAPL",
    filing_type="10-K",
    filing_date="2023-01-01"
)
```

### Quantitative Pipeline Only

```python
from sec_filing_analyzer.pipeline import QuantitativeETLPipeline

# Initialize quantitative pipeline
pipeline = QuantitativeETLPipeline(db_path="data/financial_data.duckdb")

# Process a filing
result = pipeline.process_filing(
    ticker="AAPL",
    filing_type="10-K",
    filing_date="2023-01-01"
)
```

### Search Similar Content

```python
from sec_filing_analyzer.semantic.storage import VectorStore

# Initialize vector store
vector_store = VectorStore()

# Search for similar content
results = vector_store.search(
    query_embedding=vector_store.embedding_generator.generate_embedding("revenue concentration"),
    top_k=5
)

# Display results
for result in results:
    print(f"Document ID: {result['id']}")
    print(f"Score: {result['score']}")
    print(f"Text: {result['metadata'].get('text', '')[:100]}...\n")
```

### Query Financial Data

```python
from sec_filing_analyzer.quantitative.storage import OptimizedDuckDBStore

# Initialize DuckDB store
db_store = OptimizedDuckDBStore(db_path="data/financial_data.duckdb")

# Query financial data
results = db_store.query_financial_facts(
    ticker="AAPL",
    metrics=["Revenue", "NetIncome"],
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# Display results
for result in results:
    print(f"Ticker: {result['ticker']}")
    print(f"Metric: {result['metric_name']}")
    print(f"Value: {result['value']}")
    print(f"Period: {result['period_end_date']}\n")
```

## Project Structure

```
sec_filing_analyzer/
├── config.py                # Configuration management
├── data_retrieval/          # SEC filing retrieval
│   ├── file_storage.py      # Local file storage
│   ├── filing_processor.py  # Filing processing
│   └── sec_downloader.py    # SEC EDGAR downloader
├── graphrag/                # Graph RAG components
├── pipeline/                # ETL pipeline
│   ├── etl_pipeline.py      # Main ETL pipeline
│   ├── semantic_pipeline.py # Semantic data processing pipeline
│   └── quantitative_pipeline.py # Quantitative data processing pipeline
├── semantic/                # Semantic data processing
│   ├── processing/          # Document processing
│   │   └── chunking.py      # Intelligent document chunking (1500 tokens)
│   ├── embeddings/          # Embedding generation
│   │   ├── embedding_generator.py # OpenAI embedding generation
│   │   └── parallel_embeddings.py # Parallel embedding generation
│   └── storage/             # Semantic data storage
│       └── vector_store.py  # Vector storage and search
├── quantitative/            # Quantitative data processing
│   ├── processing/          # XBRL data extraction
│   │   └── edgar_xbrl_to_duckdb.py # XBRL to DuckDB extraction
│   └── storage/             # Quantitative data storage
│       └── optimized_duckdb_store.py # DuckDB storage
├── storage/                 # Common storage implementations
│   ├── graph_store.py       # Graph database interface
│   └── interfaces.py        # Storage interfaces
└── tests/                   # Test scripts
```

## Document Processing

### Intelligent Chunking

The system uses intelligent chunking to process SEC filings:

- **Chunk Size**: 1500 tokens per chunk for optimal retrieval precision
- **Semantic Chunking**: Documents are chunked based on semantic boundaries
- **Sub-chunking**: Large chunks are automatically split into smaller sub-chunks
- **Metadata Preservation**: Each chunk maintains metadata about its source filing

### Vector Embeddings

- **OpenAI Embeddings**: Uses OpenAI's embedding models for high-quality vector representations
- **Cosine Similarity**: Search uses cosine similarity for accurate retrieval
- **Metadata Filtering**: Results can be filtered by ticker, filing type, year, etc.

## Dependencies

### Core Dependencies

- `edgar`: SEC filing retrieval and parsing (aliased as `edgartools`)
- `openai`: API access for embedding generation
- `tiktoken`: Token counting for chunking
- `rich`: Terminal UI
- `numpy`: Numerical operations for embeddings
- `pandas`: Data manipulation for filing data

### Semantic Processing

- `neo4j`: Graph database for storing relationships
- `llama-index-core`: Vector storage and search
- `llama-index-embeddings-openai`: OpenAI embeddings integration
- `faiss-cpu`: Efficient vector similarity search

### Quantitative Processing

- `duckdb`: Efficient analytical database for financial data
- `pyarrow`: Efficient data interchange format

## Scripts

### ETL Scripts

- `scripts/demo/run_nvda_etl.py`: Run the ETL pipeline for NVIDIA (demo version)
- `scripts/demo/query_revenue.py`: Query revenue data for a company (demo version)
- `scripts/test_reorganized_pipeline.py`: Test the reorganized ETL pipeline
- `scripts/test_reorganized_structure.py`: Test the reorganized directory structure

### Analysis Scripts

- `scripts/explore_vector_store.py`: Search for similar content in filings
- `scripts/direct_search.py`: Direct search using cosine similarity
- `scripts/analyze_topics.py`: Extract topics from filings
- `scripts/visualize_graph.py`: Visualize the graph database
- `scripts/query_financial_data.py`: Query financial data from DuckDB

## Logging and Visualization

### Logging Structure

The system uses a standardized logging structure:

- All logs are stored in the `data/logs/` directory
- Agent logs are stored in `data/logs/agents/`
- Workflow logs are stored in `data/logs/workflows/`
- Test logs are stored in `data/logs/tests/`
- General logs are stored in `data/logs/general/`

Logs include both plain text (`.log`) and structured JSON (`.json`) formats for easy analysis.

### Log Visualization

The SEC Filing Analyzer includes a powerful log visualization tool that provides interactive visualizations of agent and workflow logs:

- **Timeline Visualization**: View the timeline of workflow steps and operations
- **Timing Analysis**: Analyze execution times for tools, LLM calls, and workflow steps
- **Tool Call Analysis**: Examine tool usage patterns and arguments
- **LLM Interaction Analysis**: View prompts, responses, and token usage statistics

#### Usage

```bash
# Launch the Log Visualizer
python scripts/visualization/launch_log_visualizer.py

# Visualize a specific log file
python scripts/visualization/launch_log_visualizer.py --log-file data/logs/agents/FinancialAnalystAgent_20250414_125810.log
```

The visualizer supports both agent logs and workflow logs, automatically detecting the log type and displaying the appropriate visualizations. For more details, see the [Visualization README](scripts/visualization/README.md).

## Dev Hygiene & CI

This project uses several tools to maintain code quality and ensure consistent development practices:

### Code Quality Tools

- **Ruff**: Fast Python linter and formatter
- **Mypy**: Static type checker for Python
- **Bandit**: Security vulnerability scanner that identifies common security issues in Python code
- **Pytest**: Testing framework with coverage reporting to ensure code reliability
- **Pre-commit**: Git hooks to enforce checks before commits, ensuring code quality standards

For details on code quality tool configurations and exclusions, see [Code Quality Exclusions](docs/code_quality_exclusions.md).

### Setup Development Environment

1. Install dev dependencies:
   ```bash
   poetry install
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

3. Run pre-commit on all files:
   ```bash
   pre-commit run --all-files
   ```

### Running Tests

Run the tests with coverage reporting and logging:

```bash
# Using the pytest-log wrapper (recommended)
poetry run pytest-log tests/ --cov=src --cov-report=xml

# Or using the script directly
python scripts/pytest_log_wrapper.py tests/ --cov=src --cov-report=xml

# Windows PowerShell
.\scripts\run_pytest_with_logs.ps1 tests/ --cov=src --cov-report=xml

# Linux/macOS
./scripts/run_pytest_with_logs.sh tests/ --cov=src --cov-report=xml
```

This will run the tests and save the output to log files in `~/.sec_filing_analyzer_logs/pytest/`.

### Security Scanning

Run the security scanner:

```bash
bandit -r src -ll --configfile .bandit.yaml
```

### Log Collection for AI Analysis

The project automatically collects logs from code quality tools for AI analysis.

#### Automatic Log Collection

Logs are automatically generated whenever pre-commit runs. You don't need to do anything special - just use pre-commit as normal:

```bash
pre-commit run --all-files
```

If you want to generate logs without running the actual hooks (for example, to test the logging system), you can use:

**Windows (PowerShell):**
```powershell
.\scripts\log_precommit.ps1
```

**Linux/macOS:**
```bash
./scripts/log_precommit.sh
```

#### Log Locations

Logs are saved to:

**Pre-commit logs:**
- Latest run: `~/.sec_filing_analyzer_logs/latest.log`
- Timestamped logs: `~/.sec_filing_analyzer_logs/precommit_YYYYMMDD_HHMMSS.log`

**Pytest logs:**
- Latest run: `~/.sec_filing_analyzer_logs/pytest/latest.log` (text) and `~/.sec_filing_analyzer_logs/pytest/latest.xml` (JUnit XML)
- Timestamped logs: `~/.sec_filing_analyzer_logs/pytest/pytest_YYYYMMDD_HHMMSS.log` (text) and `~/.sec_filing_analyzer_logs/pytest/pytest_YYYYMMDD_HHMMSS.xml` (JUnit XML)

#### Log Features

- **Structured Format**: Each hook's output is clearly marked with a header (e.g., `===== ruff (Failed) =====`)
- **Timezone-aware Timestamps**: Logs include ISO-formatted timestamps with timezone information
- **Exit Code Preservation**: The wrapper preserves the exit code of pre-commit, ensuring that failing hooks still block commits
- **Working Directory Information**: Logs include the current working directory for context

#### Log Management

To prevent log files from accumulating indefinitely, you can use the pruning scripts:

**Windows (PowerShell):**
```powershell
.\scripts\prune_logs.ps1
```

**Linux/macOS:**
```bash
./scripts/prune_logs.sh
```

These scripts keep the 50 most recent log files and delete older ones.

#### CI Logs

GitHub Actions automatically collects logs from:
- Test runs
- Security scans
- Pre-commit checks

Logs are uploaded as artifacts with each CI run and can be downloaded from the GitHub Actions page. The logs are:

- Retained for 7 days to save storage space
- Summarized in the GitHub Actions run summary for quick review
- Tagged with the commit SHA for easy identification

### Continuous Integration

The project uses GitHub Actions for CI, which runs on every push and pull request:

- Installs Poetry and dependencies
- Runs tests with coverage reporting
- Runs Bandit for security scanning
- Stores test artifacts (coverage reports)

The CI workflow is defined in `.github/workflows/ci.yml`.

### Configuration Files

- `.pre-commit-config.yaml`: Configures pre-commit hooks
- `.bandit.yaml`: Configures security scanning rules
- `.github/workflows/ci.yml`: Defines the CI workflow
- `pyproject.toml`: Contains Poetry dependencies and tool configurations

## License

MIT
