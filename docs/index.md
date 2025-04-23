# SEC Filing Analyzer Documentation

Welcome to the SEC Filing Analyzer documentation. This documentation provides information about the SEC Filing Analyzer, a comprehensive tool for analyzing SEC filings using graph databases and vector embeddings.

## Overview

The SEC Filing Analyzer is designed to process SEC filings (10-K, 10-Q, 8-K) and extract both semantic and quantitative data. It uses a modular architecture with separate pipelines for semantic and quantitative data processing, as well as a unified pipeline that can process both types of data.

## Table of Contents

### Getting Started

- [Installation](installation.md)
- [Quick Start](quick_start.md)
- [Configuration](configuration.md)

### Architecture

- [Reorganization](reorganization.md)
- [Unified Pipeline](unified_pipeline.md)
- [Semantic Pipeline](semantic_pipeline.md)
- [Quantitative Pipeline](quantitative_pipeline.md)

### Components

- [Document Chunking](document_chunking.md)
- [Embedding Generation](embedding_generation.md)
- [Vector Storage](vector_storage.md)
- [Graph Storage](graph_storage.md)
- [XBRL Extraction](xbrl_extraction.md)
- [DuckDB Storage](duckdb_storage.md)
- [Logging](logging.md)

### Agent Architecture

- [Agent Parameters and Phases](agent_parameters.md)
- [Agent Loop and Iteration Parameters](agent_loop.md)
- [Tool Ledger](tool_ledger.md)
- [Planning Capability](planning_capability.md)
- [Parameter Completion](parameter_completion.md)
- [Plan-Step ↔ Tool Contract](plan_step_tool_contract.md)
- [Tool Parameter Models](tool_parameter_models.md)
- [Error Handling](error_handling.md)

### Usage

- [Processing Filings](processing_filings.md)
- [Semantic Search](semantic_search.md)
- [Financial Analysis](financial_analysis.md)
- [Time-Series Analysis](time_series_analysis.md)

### Development

- [Contributing](contributing.md)
- [Testing](testing.md)
- [Code Style](code_style.md)

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
