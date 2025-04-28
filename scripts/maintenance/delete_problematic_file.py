#!/usr/bin/env python
"""
Delete a problematic file with a long name.
"""

import os
from pathlib import Path

# Define the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Define the problematic file name
PROBLEMATIC_FILE = "rc.agents.coordinator import FinancialDiligenceCoordinator; coordinator = FinancialDiligenceCoordinator(); print('Successfully initialized coordinator with config-based specialized agents')"

# Delete the file
file_path = ROOT_DIR / PROBLEMATIC_FILE
if file_path.exists():
    print(f"Deleting problematic file: {PROBLEMATIC_FILE}")
    os.remove(file_path)
    print("File deleted successfully")
else:
    print(f"File not found: {PROBLEMATIC_FILE}")

print("Done")
