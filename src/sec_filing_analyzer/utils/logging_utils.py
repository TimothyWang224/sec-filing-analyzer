"""
Logging Utilities Module

This module provides enhanced logging functionality for the SEC Filing Analyzer.
"""

import os
import logging
import traceback
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Create a custom logger for embedding errors
embedding_logger = logging.getLogger('embedding_errors')

# Ensure the logger doesn't propagate to the root logger
embedding_logger.propagate = False

def setup_logging(log_dir: Optional[Path] = None) -> None:
    """Set up enhanced logging for the application.

    Args:
        log_dir: Directory to store log files (default: data/logs)
    """
    if log_dir is None:
        log_dir = Path('data/logs')

    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a file handler for embedding errors
    embedding_log_file = log_dir / f'embedding_errors_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.FileHandler(embedding_log_file)
    file_handler.setLevel(logging.ERROR)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    embedding_logger.setLevel(logging.ERROR)

    # Remove existing handlers to avoid duplicates
    for handler in embedding_logger.handlers[:]:  # Make a copy of the list
        embedding_logger.removeHandler(handler)

    embedding_logger.addHandler(file_handler)

    # Also log to console for critical errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)
    console_handler.setFormatter(formatter)
    embedding_logger.addHandler(console_handler)

    # Create a summary file for embedding errors
    summary_file = log_dir / 'embedding_errors_summary.json'
    if not summary_file.exists():
        with open(summary_file, 'w') as f:
            json.dump([], f)

def log_embedding_error(
    error: Exception,
    filing_id: str,
    company: str,
    filing_type: str,
    batch_index: Optional[int] = None,
    chunk_count: Optional[int] = None
) -> None:
    """Log detailed information about embedding errors.

    Args:
        error: The exception that occurred
        filing_id: The filing ID (accession number)
        company: The company name or ticker
        filing_type: The type of filing (e.g., 10-K, 10-Q)
        batch_index: Optional batch index for batch processing
        chunk_count: Optional count of chunks being processed
    """
    # Get the full stack trace
    stack_trace = traceback.format_exc()

    # Create a detailed error message
    error_message = f"Embedding error for {company} {filing_type} (ID: {filing_id}): {str(error)}"
    if batch_index is not None:
        error_message += f" - Batch: {batch_index}"
    if chunk_count is not None:
        error_message += f" - Chunks: {chunk_count}"

    # Log the error with stack trace
    embedding_logger.error(f"{error_message}\n{stack_trace}")

    # Add to summary file
    log_dir = Path('data/logs')
    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    summary_file = log_dir / 'embedding_errors_summary.json'

    try:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        summary = []

    # Add new error to summary
    summary.append({
        'timestamp': datetime.now().isoformat(),
        'filing_id': filing_id,
        'company': company,
        'filing_type': filing_type,
        'error': str(error),
        'batch_index': batch_index,
        'chunk_count': chunk_count
    })

    # Write updated summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

def generate_embedding_error_report() -> str:
    """Generate a report of embedding errors.

    Returns:
        A formatted report of embedding errors
    """
    log_dir = Path('data/logs')
    summary_file = log_dir / 'embedding_errors_summary.json'

    try:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return "No embedding errors found or summary file not available."

    if not summary:
        return "No embedding errors recorded."

    # Group errors by company
    errors_by_company = {}
    error_types = {}

    for error in summary:
        company = error['company']
        if company not in errors_by_company:
            errors_by_company[company] = []
        errors_by_company[company].append(error)

        # Track error types
        error_msg = error['error']
        error_type = error_msg.split(':', 1)[0] if ':' in error_msg else error_msg
        if error_type not in error_types:
            error_types[error_type] = 0
        error_types[error_type] += 1

    # Format the report
    report = "Embedding Error Report\n"
    report += "=====================\n\n"
    report += f"Total errors: {len(summary)}\n\n"

    # Error type summary
    report += "Error Types:\n"
    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        report += f"  - {error_type}: {count} occurrences\n"
    report += "\n"

    # Company-specific errors
    report += "Errors by Company:\n"
    for company, errors in sorted(errors_by_company.items(), key=lambda x: len(x[1]), reverse=True):
        report += f"{company}: {len(errors)} errors\n"
        # Show the first 5 errors for each company
        for error in errors[:5]:
            report += f"  - {error['filing_type']} (ID: {error['filing_id']}): {error['error']}\n"
        if len(errors) > 5:
            report += f"  - ... and {len(errors) - 5} more errors\n"
        report += "\n"

    return report
