# SEC Filing Analyzer - Streamlit Application

This directory contains the Streamlit application for the SEC Filing Analyzer, providing a user-friendly interface for the ETL pipeline and agent workflow.

## Features

- **Dashboard**: Overview of system status and quick access to key features
- **ETL Pipeline**: Configure and run the ETL pipeline for SEC filings
- **Agent Workflow**: Interact with intelligent agents to analyze SEC filings
- **Data Explorer**: Explore the data extracted from SEC filings
- **Configuration**: Configure the SEC Filing Analyzer system

## Getting Started

### Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- Streamlit
- SEC Filing Analyzer dependencies

### Installation

1. Install the SEC Filing Analyzer package:
   ```bash
   poetry install
   ```

2. Set up the required environment variables in a `.env` file:
   ```
   EDGAR_IDENTITY=your_email@example.com
   OPENAI_API_KEY=your_openai_api_key
   ```

### Running the Application

#### Using the Launcher Script

The easiest way to run the application is using the launcher script:

```bash
# On Windows
launch_app.bat

# On Linux/Mac
./launch_app.sh
```

#### Running Directly

You can also run the application directly using Python:

```bash
poetry run python src/streamlit_app/run_app.py
```

Or using Streamlit:

```bash
poetry run streamlit run src/streamlit_app/app.py
```

## Application Structure

- `app.py`: Main entry point for the Streamlit application
- `pages/`: Directory containing the individual pages of the application
  - `etl_pipeline.py`: ETL Pipeline page
  - `agent_workflow.py`: Agent Workflow page
  - `data_explorer.py`: Data Explorer page
  - `configuration.py`: Configuration page
- `run_app.py`: Script to launch the Streamlit application

## Configuration

The application uses the SEC Filing Analyzer configuration system, which can be configured through:

1. The Configuration page in the application
2. Environment variables
3. Configuration files in `data/config/`

## Troubleshooting

If you encounter issues:

1. Check that all dependencies are installed: `poetry install`
2. Verify that environment variables are set correctly
3. Check the logs in `data/logs/`
4. Make sure the required databases (DuckDB, Neo4j) are accessible

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
