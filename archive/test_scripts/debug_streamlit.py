"""
Debug Streamlit App

This script provides detailed debugging information when launching the Streamlit app.
"""

import os
import sys
import socket
import traceback
from pathlib import Path

def check_port(port):
    """Check if a port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result != 0  # True if port is available

def check_dependencies():
    """Check if all required dependencies are installed."""
    dependencies = [
        "streamlit",
        "pandas",
        "plotly",
        "duckdb"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} is installed")
        except ImportError:
            missing.append(dep)
            print(f"✗ {dep} is NOT installed")
    
    return missing

def check_python_path():
    """Check Python path."""
    print("\nPython Path:")
    for path in sys.path:
        print(f"  - {path}")

def main():
    """Run diagnostics and launch Streamlit."""
    print("=== SEC Filing Analyzer Streamlit App Diagnostics ===\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in the project root
    expected_files = ["pyproject.toml", "README.md"]
    for file in expected_files:
        if os.path.exists(file):
            print(f"✓ Found {file} in current directory")
        else:
            print(f"✗ Could not find {file} in current directory")
    
    # Check dependencies
    print("\nChecking dependencies:")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_deps)}")
    
    # Check Python path
    check_python_path()
    
    # Check app file
    app_path = Path("src") / "streamlit_app" / "app.py"
    if app_path.exists():
        print(f"\n✓ Found app file at {app_path}")
    else:
        print(f"\n✗ Could not find app file at {app_path}")
        print("Make sure you're running this script from the project root directory.")
        return
    
    # Check port availability
    port = 8501
    if check_port(port):
        print(f"\n✓ Port {port} is available")
    else:
        print(f"\n✗ Port {port} is already in use")
        print("This could be causing the connection issues.")
        
        # Try to find an available port
        for test_port in range(8502, 8510):
            if check_port(test_port):
                print(f"Port {test_port} is available. Try using this port instead.")
                break
    
    # Try to import the config module
    print("\nTrying to import configuration module:")
    try:
        sys.path.insert(0, str(Path(current_dir)))
        from sec_filing_analyzer.config import ConfigProvider
        print("✓ Successfully imported ConfigProvider")
    except ImportError as e:
        print(f"✗ Error importing ConfigProvider: {e}")
        traceback.print_exc()
    
    # Ask user if they want to try launching Streamlit directly
    print("\n=== Diagnostic Complete ===")
    print("\nWould you like to try launching Streamlit directly? (y/n)")
    choice = input().lower()
    
    if choice == 'y':
        print("\nLaunching Streamlit directly...")
        os.environ["STREAMLIT_SERVER_PORT"] = str(port)
        os.system(f"streamlit run {app_path}")
    else:
        print("\nExiting without launching Streamlit.")

if __name__ == "__main__":
    main()
