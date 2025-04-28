# Contributing to SEC Filing Analyzer

Thank you for your interest in contributing to the SEC Filing Analyzer project! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Installation Options

We use Poetry's dependency groups to organize our dependencies. This allows for more efficient installations depending on your needs.

#### Full Installation (All Dependencies)

```bash
poetry install --with dev,data,vector,nlp,tools
```

#### Minimal Installation (Core Dependencies Only)

If you only need the core functionality:

```bash
poetry install
```

#### Development Installation (For Code Contributions)

If you're contributing code and need development tools but not all the heavy data dependencies:

```bash
poetry install --with dev
```

#### Linting-Only Installation (For Quick Fixes)

If you're only making small changes and need to run linters:

```bash
poetry install --with dev --without data,vector,nlp,tools
```

This is the fastest installation option and is recommended for contributors who are only making small changes or documentation updates.

### Dependency Groups

The project's dependencies are organized into the following groups:

- **Core**: Essential dependencies for basic functionality
- **data**: Data visualization and UI dependencies (Streamlit, Plotly, etc.)
- **vector**: Vector database dependencies (FAISS, ChromaDB, etc.)
- **nlp**: Natural language processing dependencies (Spacy, etc.)
- **dev**: Development tools (pytest, ruff, mypy, etc.)
- **tools**: Data exploration tools (Jupyter, pandas, etc.)

## Development Workflow

### Code Style

We use the following tools to maintain code quality:

- **Ruff**: For linting and code style checking
- **Black**: For code formatting
- **MyPy**: For type checking

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. Install the pre-commit hooks with:

```bash
pre-commit install
```

You can run the pre-commit hooks manually with:

```bash
pre-commit run --all-files
```

### Running Tests

Run the tests with pytest:

```bash
poetry run pytest
```

Or with the log wrapper to capture detailed logs:

```bash
poetry run pytest-log
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linters
5. Submit a pull request

## License

By contributing to this project, you agree that your contributions will be licensed under the project's license.
