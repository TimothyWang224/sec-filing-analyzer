# Archived Scripts

This directory contains archived scripts that were previously in the root directory of the project. These scripts were used for one-time tasks, fixes, or tests and are no longer needed for regular operation of the application.

## Organization

Scripts are organized in timestamped directories (YYYYMMDD_HHMMSS) to preserve the history of when they were archived.

## Contents

The archived scripts typically include:

- One-time database fixes and schema updates
- Temporary test scripts
- Diagnostic tools used during development
- Scripts used to fix specific issues that have been resolved

## Usage

These scripts are kept for reference purposes only and should not be used in production. If you need to reuse any functionality from these scripts, consider creating a new script in the appropriate directory under `scripts/`.

## Archiving Process

Scripts were archived using the `organize_root_directory.py` script, which:
1. Moved useful tools to the `scripts/` directory
2. Archived temporary test/fix scripts to `archive/scripts/`
3. Kept only essential files in the root directory
