# Running the SEC Filing Analyzer

This document provides instructions for running the SEC Filing Analyzer application.

## Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)
- Git (for version control)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sec-filing-analyzer.git
   cd sec-filing-analyzer
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your EDGAR identity for SEC API access:
     ```
     EDGAR_IDENTITY="Your Name (your.email@example.com)"
     ```

## Running the Application

### On Windows

Simply double-click the `run_app.bat` file in the project root directory, or run it from the command line:

```
run_app.bat
```

### On macOS/Linux

Run the Python launcher script:

```
python run_app.py
```

Or use Poetry directly:

```
poetry run python src/streamlit_app/run_app.py
```

## Accessing the Application

Once the application is running, it will automatically open in your default web browser at:

```
http://localhost:8501
```

If port 8501 is already in use, the application will automatically find an available port and display the URL in the console.

## Troubleshooting

If you encounter any issues:

1. Check that Poetry is installed and in your PATH
2. Ensure all dependencies are installed: `poetry install`
3. Verify that your `.env` file contains the required EDGAR_IDENTITY
4. Check the console output for any error messages

For more detailed information, see the main README.md file.
