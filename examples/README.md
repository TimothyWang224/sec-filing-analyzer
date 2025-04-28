# SEC Filing Analyzer Demo

This directory contains demo applications for the SEC Filing Analyzer.

## Overview

The demos provide simplified versions of the full SEC Filing Analyzer, focusing on a single agent with a limited set of tools. This makes it easier to demonstrate the core functionality without the complexity of the full hierarchical planner.

## Available Demos

### CLI Demo

A simple command-line interface for interacting with the SEC Filing Analyzer:

```bash
# Run the CLI demo
poetry run chat-demo

# Or run the script directly
poetry run python examples/run_chat_demo.py
```

### Streamlit Demo

A web-based interface for interacting with the SEC Filing Analyzer:

```bash
# Run the Streamlit demo
poetry run chat-demo-web

# Or run the script directly
poetry run streamlit run examples/streamlit_demo.py
```

## Features

Both demos provide access to the following tools:

- **VectorSearchTool**: Search for information in SEC filings
- **FinancialFactsTool**: Query financial metrics and facts
- **SECFilingsTool**: Get information about companies and filings

## Installation

To install the dependencies for the demos:

```bash
# For the CLI demo only
poetry install --with dev

# For the Streamlit demo
poetry install --with dev,demo
```

## Demo Assets

The demo uses a simplified vector store located at `data/demo_assets/vector_demo`. This contains a subset of SEC filings for demonstration purposes.

## Implementation Details

The demos consist of the following components:

1. **SimpleChatAgent**: A simplified agent implementation that uses a limited set of tools
2. **run_chat_demo.py**: The CLI demo script
3. **streamlit_demo.py**: The Streamlit demo script
4. **Tools**: Wrappers around the core SEC Filing Analyzer tools

The demos leverage the existing codebase but bypass the complex hierarchical planner in favor of a simpler approach.
