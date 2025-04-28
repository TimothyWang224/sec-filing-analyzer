# SEC Filing Analyzer Demo

This directory contains scripts for demonstrating the SEC Filing Analyzer's capabilities in a clean, concise way.

## Demo Options

### 1. Command-Line Demo

The command-line demo shows that the project installs cleanly, ingests raw SEC filings, builds embeddings, and answers concrete finance questions—all without venturing into the trickier multi-agent graph.

### 2. Streamlit Demo (New!)

For a more visual and interactive experience, we've added a Streamlit demo that provides a sleek, user-friendly interface for exploring the SEC Filing Analyzer's capabilities. This is perfect for recording demos with tools like Loom.

To run the Streamlit demo:

```bash
# From the project root
poetry run streamlit run scripts/demo/streamlit_demo.py

# Or use the convenience scripts
./run_demo.sh  # On macOS/Linux
run_demo.bat   # On Windows
```

The Streamlit demo provides:
- A clean, modern UI for interacting with the SEC Filing Analyzer
- Interactive controls for running the ETL process
- Visualizations of financial data
- Quick revenue lookup functionality

This is ideal for showcasing the project's capabilities in a visually appealing way.

### Step 1: Clone & Install

```bash
git clone https://github.com/TimothyWang224/sec-filing-analyzer.git
cd sec-filing-analyzer
poetry install --no-root
```

### Step 2: One-click ETL

```bash
export OPENAI_API_KEY=sk-...
poetry run python scripts/demo/run_nvda_etl.py --ticker NVDA --years 2023 2024
```

This will:
- Download real NVIDIA 10-K filings from SEC EDGAR
- Parse the filings and extract financial data
- Build a vector store
- Write financial data to DuckDB

#### Using Synthetic Data (Optional)

For testing or when SEC EDGAR is unavailable, you can use synthetic data:

```bash
# Using command-line flag
poetry run python scripts/demo/run_nvda_etl.py --ticker NVDA --years 2023 2024 --test-mode

# Or using environment variable
TEST_MODE=True poetry run python scripts/demo/run_nvda_etl.py --ticker NVDA --years 2023 2024
```

### Step 3: Ask a Finance Question

```bash
poetry run python scripts/demo/query_revenue.py --ticker NVDA --year 2023
```

This will return:
```
FY-2023 revenue = $26.97 B (source: Form 10-K, p. 12)
```

## What Happens Under the Hood

1. **Download 10-K / 10-Q from EDGAR**
   - The system connects to the SEC's EDGAR database
   - It retrieves the filings for the specified company and years

2. **Parse XBRL → Structured Tables (DuckDB)**
   - The system extracts XBRL data from the filings
   - It stores the data in a structured format in DuckDB

3. **Embed & Store Sections in a Local Vector DB**
   - The system chunks the filings into sections
   - It generates embeddings for each section
   - It stores the embeddings in a local vector database

4. **RAG-prompt GPT-4 to Answer and Cite**
   - The system uses the vector database to find relevant sections
   - It prompts GPT-4 with the relevant sections to answer questions
   - It provides citations to the source filings

## Why This Demo is "Good Enough"

- **Recruiter proof point:** Even non-dev finance head-hunters strongly prefer candidates who *show* a working artifact (46% higher callback rate). Using real SEC filings adds authenticity and credibility.
- **Avoids sunk-cost fallacy:** You invest <1 day, then shift deep-work time to your commercial MVP instead of chasing diminishing debug returns.
- **Portable talking piece:** The demo doubles as a design-partner teaser when you pitch the PE-facing tool.
- **Authenticity & credibility:** Using real SEC filings demonstrates that your pipeline works with production-grade text, including all the subtle footnotes and XBRL quirks that synthetic data might miss.
- **Reproducibility:** Anyone watching the demo can clone the repo and replicate it exactly, with the option to fall back to synthetic data if needed.
