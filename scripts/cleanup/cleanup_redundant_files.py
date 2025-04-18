#!/usr/bin/env python
"""
Cleanup Redundant Files

This script helps clean up redundant files in the project.
It identifies redundant launcher scripts and batch files and suggests which ones to keep.
"""

import os
import argparse
from pathlib import Path

# Files to keep
KEEP_FILES = [
    "run_app.bat",
    "run_app.py",
    "src/streamlit_app/run_app.py",
]

# Patterns to identify redundant files
REDUNDANT_PATTERNS = [
    "launch_app.bat",
    "launch_app.py",
    "run_streamlit.bat",
    "run_streamlit.py",
    "start_app.bat",
    "start_app.py",
]

def find_redundant_files(root_dir="."):
    """Find redundant files in the project."""
    root_path = Path(root_dir).resolve()
    redundant_files = []
    
    # Walk through the directory tree
    for path in root_path.glob("**/*"):
        if path.is_file():
            # Check if the file matches any redundant pattern
            if any(pattern in path.name for pattern in REDUNDANT_PATTERNS):
                # Check if it's not in the keep list
                if not any(str(path.relative_to(root_path)) == keep_file for keep_file in KEEP_FILES):
                    redundant_files.append(path)
    
    return redundant_files

def suggest_cleanup(redundant_files):
    """Suggest cleanup actions for redundant files."""
    if not redundant_files:
        print("No redundant files found.")
        return
    
    print(f"Found {len(redundant_files)} redundant files:")
    for file in redundant_files:
        print(f"  - {file}")
    
    print("\nRecommended actions:")
    print("1. Keep the following files:")
    for file in KEEP_FILES:
        print(f"  - {file}")
    
    print("\n2. Remove the redundant files listed above.")
    print("   You can use the following command to remove them:")
    print("   python scripts/cleanup/cleanup_redundant_files.py --remove")

def remove_redundant_files(redundant_files):
    """Remove redundant files."""
    if not redundant_files:
        print("No redundant files to remove.")
        return
    
    for file in redundant_files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cleanup redundant files in the project.")
    parser.add_argument("--remove", action="store_true", help="Remove redundant files")
    args = parser.parse_args()
    
    redundant_files = find_redundant_files()
    
    if args.remove:
        remove_redundant_files(redundant_files)
    else:
        suggest_cleanup(redundant_files)

if __name__ == "__main__":
    main()
