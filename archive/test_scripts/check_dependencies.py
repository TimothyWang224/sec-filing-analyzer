"""
Check Dependencies

This script checks for missing dependencies that might be causing issues.
"""

import importlib
import os
import sys
from pathlib import Path


def check_import(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError as e:
        return False, str(e)


def main():
    """Check dependencies."""
    print("Checking dependencies...")

    # Basic dependencies
    basic_deps = ["streamlit", "pandas", "plotly", "duckdb", "faiss", "psutil"]

    print("\nBasic Dependencies:")
    for dep in basic_deps:
        result = check_import(dep)
        if result is True:
            print(f"✓ {dep}")
        else:
            print(f"✗ {dep}: {result[1]}")

    # Check for sec_filing_analyzer package
    print("\nSEC Filing Analyzer Package:")
    result = check_import("sec_filing_analyzer")
    if result is True:
        print("✓ sec_filing_analyzer")

        # Check submodules
        submodules = [
            "sec_filing_analyzer.config",
            "sec_filing_analyzer.storage",
            "sec_filing_analyzer.llm.llm_config",
        ]

        print("\nSEC Filing Analyzer Submodules:")
        for submodule in submodules:
            result = check_import(submodule)
            if result is True:
                print(f"✓ {submodule}")
            else:
                print(f"✗ {submodule}: {result[1]}")
    else:
        print(f"✗ sec_filing_analyzer: {result[1]}")

    # Check Python path
    print("\nPython Path:")
    for path in sys.path:
        print(f"- {path}")

    # Check current directory
    print(f"\nCurrent Directory: {os.getcwd()}")

    # Check if pyproject.toml exists
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        print(f"✓ pyproject.toml found at {pyproject_path.absolute()}")
    else:
        print(f"✗ pyproject.toml not found at {pyproject_path.absolute()}")

    # Check if src directory exists
    src_path = Path("src")
    if src_path.exists():
        print(f"✓ src directory found at {src_path.absolute()}")
    else:
        print(f"✗ src directory not found at {src_path.absolute()}")


if __name__ == "__main__":
    main()
