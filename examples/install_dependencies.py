#!/usr/bin/env python
"""
Install dependencies required for the demo.

This script checks if the required dependencies are installed and installs them if needed.
"""

import importlib
import logging
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# List of required packages
REQUIRED_PACKAGES = [
    "python-dotenv",
    "streamlit",
    "plotly",
    "pandas",
    "duckdb",
]


def check_package(package_name):
    """Check if a package is installed."""
    try:
        # Handle special cases
        if package_name == "python-dotenv":
            module_name = "dotenv"
        else:
            module_name = package_name.replace("-", "_")

        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def install_package(package_name):
    """Install a package using pip."""
    logger.info(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False


def main():
    """Check and install required packages."""
    logger.info("Checking required packages...")

    missing_packages = []
    for package in REQUIRED_PACKAGES:
        if not check_package(package):
            logger.warning(f"{package} is not installed")
            missing_packages.append(package)

    if not missing_packages:
        logger.info("All required packages are installed")
        return 0

    logger.info(f"Installing {len(missing_packages)} missing packages...")

    success_count = 0
    for package in missing_packages:
        if install_package(package):
            success_count += 1

    if success_count == len(missing_packages):
        logger.info("All missing packages were successfully installed")
        return 0
    else:
        logger.error(f"Failed to install {len(missing_packages) - success_count} packages")
        return 1


if __name__ == "__main__":
    sys.exit(main())
