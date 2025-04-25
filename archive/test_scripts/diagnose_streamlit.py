"""
Diagnose Streamlit Issues

This script checks for common issues with Streamlit and provides diagnostic information.
"""

import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path


def check_import(module_name):
    """Check if a module can be imported."""
    try:
        module = importlib.import_module(module_name)
        return True, module
    except ImportError as e:
        return False, str(e)


def get_module_version(module):
    """Get the version of a module."""
    try:
        return module.__version__
    except AttributeError:
        return "Unknown"


def check_streamlit_version():
    """Check the Streamlit version."""
    success, streamlit = check_import("streamlit")
    if success:
        return get_module_version(streamlit)
    else:
        return f"Error: {streamlit}"


def check_python_version():
    """Check the Python version."""
    return sys.version


def check_os_info():
    """Check the OS information."""
    return platform.platform()


def check_dependencies():
    """Check dependencies."""
    dependencies = ["streamlit", "pandas", "numpy", "plotly", "duckdb", "faiss", "psutil", "neo4j"]

    results = {}
    for dep in dependencies:
        success, module = check_import(dep)
        if success:
            results[dep] = get_module_version(module)
        else:
            results[dep] = f"Error: {module}"

    return results


def check_streamlit_config():
    """Check the Streamlit configuration."""
    config_path = Path.home() / ".streamlit" / "config.toml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return f.read()
    else:
        return "No Streamlit config file found."


def check_environment_variables():
    """Check environment variables."""
    env_vars = [
        "PYTHONPATH",
        "PATH",
        "STREAMLIT_SERVER_PORT",
        "STREAMLIT_SERVER_HEADLESS",
        "STREAMLIT_SERVER_ADDRESS",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS",
    ]

    results = {}
    for var in env_vars:
        results[var] = os.environ.get(var, "Not set")

    return results


def check_file_permissions():
    """Check file permissions."""
    paths = [
        "src/streamlit_app/pages/data_explorer.py",
        "src/streamlit_app/pages/data_explorer_debug.py",
        "src/streamlit_app/pages/minimal_explorer.py",
    ]

    results = {}
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    f.read(1)  # Try to read one byte
                results[path] = "Readable"
            except Exception as e:
                results[path] = f"Error: {str(e)}"
        else:
            results[path] = "File not found"

    return results


def check_streamlit_process():
    """Check if Streamlit is already running."""
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "name"]):
            if "streamlit" in proc.info["name"].lower():
                return f"Streamlit is running with PID {proc.info['pid']}"
        return "No Streamlit process found"
    except ImportError:
        return "psutil not available, cannot check for Streamlit process"


def main():
    """Main function."""
    print("Diagnosing Streamlit issues...\n")

    print(f"Python version: {check_python_version()}")
    print(f"OS information: {check_os_info()}")
    print(f"Streamlit version: {check_streamlit_version()}")
    print(f"Streamlit process: {check_streamlit_process()}")

    print("\nDependencies:")
    dependencies = check_dependencies()
    for dep, version in dependencies.items():
        print(f"  {dep}: {version}")

    print("\nEnvironment variables:")
    env_vars = check_environment_variables()
    for var, value in env_vars.items():
        print(f"  {var}: {value}")

    print("\nFile permissions:")
    permissions = check_file_permissions()
    for path, status in permissions.items():
        print(f"  {path}: {status}")

    print("\nStreamlit config:")
    print(check_streamlit_config())

    print("\nCurrent directory:")
    print(f"  {os.getcwd()}")

    print("\nPython path:")
    for path in sys.path:
        print(f"  {path}")

    print("\nDiagnosis complete.")


if __name__ == "__main__":
    main()
