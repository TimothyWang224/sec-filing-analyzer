"""
Organize the root directory by moving or archiving Python files.

This script:
1. Moves useful tools to the scripts/ directory
2. Archives temporary test/fix scripts to archive/scripts/
3. Keeps only essential files in the root directory
"""

import datetime
import os
import shutil
from pathlib import Path

# Files to move to scripts directory
FILES_TO_MOVE = ["duckdb_explorer.py", "test_sync_mismatches.py"]

# Files to archive (temporary test/fix scripts)
FILES_TO_ARCHIVE = [
    "check_db.py",
    "check_mapping.py",
    "fix_db_schema.py",
    "fix_db_schema_complete.py",
    "fix_db_schema_final.py",
    "fix_sync_manager.py",
    "test_sync_after_fix.py",
    "test_sync_manager.py",
    "test_sync_manager_functions.py",
    "add_nvda_to_db.py",
]

# Files to keep in root directory
FILES_TO_KEEP = [
    "run_app.py",
    "organize_root_directory.py",  # Keep this script
]


def main():
    """Execute the cleanup and organization."""
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent

    # Create scripts subdirectories if they don't exist
    scripts_dir = root_dir / "scripts"
    scripts_tools_dir = scripts_dir / "tools"
    scripts_tests_dir = scripts_dir / "tests"

    os.makedirs(scripts_tools_dir, exist_ok=True)
    os.makedirs(scripts_tests_dir, exist_ok=True)

    # Create archive directory if it doesn't exist
    archive_dir = root_dir / "archive" / "scripts"
    archive_timestamp_dir = archive_dir / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if archive directory already exists
    if not archive_dir.exists():
        os.makedirs(archive_dir, exist_ok=True)

    # Create timestamped archive directory
    os.makedirs(archive_timestamp_dir, exist_ok=True)

    # Process files to move
    for filename in FILES_TO_MOVE:
        src_path = root_dir / filename
        if not src_path.exists():
            print(f"Warning: {filename} not found in root directory")
            continue

        # Determine destination based on file type
        if filename == "duckdb_explorer.py":
            dst_path = scripts_tools_dir / filename
        elif filename.startswith("test_"):
            dst_path = scripts_tests_dir / filename
        else:
            dst_path = scripts_dir / filename

        # Move the file
        try:
            shutil.copy2(src_path, dst_path)
            os.remove(src_path)
            print(f"Moved {filename} to {dst_path.relative_to(root_dir)}")
        except Exception as e:
            print(f"Error moving {filename}: {e}")

    # Process files to archive
    for filename in FILES_TO_ARCHIVE:
        src_path = root_dir / filename
        if not src_path.exists():
            print(f"Warning: {filename} not found in root directory")
            continue

        # Archive the file
        dst_path = archive_timestamp_dir / filename
        try:
            shutil.copy2(src_path, dst_path)
            os.remove(src_path)
            print(f"Archived {filename} to {dst_path.relative_to(root_dir)}")
        except Exception as e:
            print(f"Error archiving {filename}: {e}")

    # Check for any remaining .py files in root directory
    for py_file in root_dir.glob("*.py"):
        if (
            py_file.name not in FILES_TO_KEEP
            and py_file.name not in FILES_TO_MOVE
            and py_file.name not in FILES_TO_ARCHIVE
        ):
            print(f"Warning: Unhandled Python file in root directory: {py_file.name}")

    print("\nCleanup and organization completed!")
    print(f"- Files moved to scripts directory: {len(FILES_TO_MOVE)}")
    print(f"- Files archived: {len(FILES_TO_ARCHIVE)}")
    print(f"- Archive location: {archive_timestamp_dir.relative_to(root_dir)}")


if __name__ == "__main__":
    main()
