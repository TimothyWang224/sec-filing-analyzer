"""
ETL Logging Module

This module provides enhanced logging functionality specifically for ETL processes
in the SEC Filing Analyzer. It includes structured logging, performance metrics,
and rate limit monitoring.
"""

import json
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import ETLConfig to get the database path
from ..config import ConfigProvider, ETLConfig

# Create a custom logger for ETL processes
etl_logger = logging.getLogger("etl_pipeline")

# Ensure the logger doesn't propagate to the root logger
etl_logger.propagate = False

# Dictionary to store run statistics
_run_stats = {}
_run_stats_lock = threading.Lock()


class ETLLogFormatter(logging.Formatter):
    """Custom formatter for ETL logs that includes additional context."""

    def format(self, record):
        """Format the log record with additional context."""
        # Add ISO timestamp
        record.iso_timestamp = datetime.fromtimestamp(record.created).isoformat()

        # Add run_id if available
        if hasattr(record, "run_id"):
            record.run_id = record.run_id
        else:
            record.run_id = "unknown"

        # Format the message
        return super().format(record)


def get_etl_log_dir() -> Path:
    """Get the ETL log directory path.

    Returns:
        Path to the ETL log directory
    """
    base_log_dir = Path("data/logs/etl")
    base_log_dir.mkdir(parents=True, exist_ok=True)
    return base_log_dir


def setup_etl_logging(log_level: int = logging.INFO) -> None:
    """Set up enhanced logging for ETL processes.

    Args:
        log_level: Logging level (default: INFO)
    """
    log_dir = get_etl_log_dir()

    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a file handler for ETL logs
    today = datetime.now().strftime("%Y%m%d")
    etl_log_file = log_dir / f"etl_{today}.log"
    file_handler = logging.FileHandler(etl_log_file)
    file_handler.setLevel(log_level)

    # Create a formatter
    formatter = ETLLogFormatter("%(iso_timestamp)s - [%(run_id)s] - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    etl_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in etl_logger.handlers[:]:
        etl_logger.removeHandler(handler)

    etl_logger.addHandler(file_handler)

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    etl_logger.addHandler(console_handler)

    # Create a summary file for ETL runs
    summary_file = log_dir / "etl_runs_summary.json"
    if not summary_file.exists():
        with open(summary_file, "w") as f:
            json.dump([], f)


def generate_run_id() -> str:
    """Generate a unique run ID for an ETL process.

    Returns:
        A unique run ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"etl_{timestamp}"


def log_etl_start(run_id: str, parameters: Dict[str, Any], description: Optional[str] = None) -> None:
    """Log the start of an ETL process.

    Args:
        run_id: Unique identifier for this ETL run
        parameters: Parameters used for this ETL run
        description: Optional description of this ETL run
    """
    # Get database path from ETLConfig
    try:
        etl_config = ConfigProvider.get_config(ETLConfig)
        db_path = etl_config.db_path
    except Exception:
        db_path = "data/db_backup/improved_financial_data.duckdb"

    # Initialize run statistics
    with _run_stats_lock:
        _run_stats[run_id] = {
            "start_time": time.time(),
            "end_time": None,
            "status": "running",
            "parameters": parameters,
            "description": description,
            "database_path": db_path,
            "companies_processed": 0,
            "filings_processed": 0,
            "filings_failed": 0,
            "filings_skipped": 0,
            "api_calls": 0,
            "api_errors": 0,
            "rate_limit_history": [],
            "embedding_stats": {
                "total_tokens": 0,
                "total_chunks": 0,
                "fallback_count": 0,
            },
            "phase_timings": {},
            "errors": [],
        }

    # Log start message
    extra = {"run_id": run_id}
    etl_logger.info(f"Starting ETL process with parameters: {json.dumps(parameters)}", extra=extra)

    # Add to summary file
    log_dir = get_etl_log_dir()
    summary_file = log_dir / "etl_runs_summary.json"

    try:
        with open(summary_file, "r") as f:
            summary = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        summary = []

    # Add new run to summary
    summary.append(
        {
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "status": "running",
            "parameters": parameters,
            "description": description,
        }
    )

    # Write updated summary
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


def log_etl_end(run_id: str, status: str = "completed", error: Optional[str] = None) -> None:
    """Log the end of an ETL process.

    Args:
        run_id: Unique identifier for this ETL run
        status: Status of the ETL run (completed, failed, etc.)
        error: Optional error message if the ETL run failed
    """
    end_time = time.time()

    # Update run statistics
    with _run_stats_lock:
        if run_id in _run_stats:
            _run_stats[run_id]["end_time"] = end_time
            _run_stats[run_id]["status"] = status
            if error:
                _run_stats[run_id]["errors"].append({"timestamp": datetime.now().isoformat(), "message": error})

            # Calculate duration
            start_time = _run_stats[run_id]["start_time"]
            duration = end_time - start_time

            # Log end message
            extra = {"run_id": run_id}
            etl_logger.info(
                f"ETL process {status} in {duration:.2f} seconds. "
                f"Processed {_run_stats[run_id]['filings_processed']} filings, "
                f"failed {_run_stats[run_id]['filings_failed']}, "
                f"skipped {_run_stats[run_id]['filings_skipped']}.",
                extra=extra,
            )

            # Save detailed run statistics
            stats_file = get_etl_log_dir() / f"{run_id}_stats.json"
            with open(stats_file, "w") as f:
                json.dump(_run_stats[run_id], f, indent=2)
        else:
            # Log warning if run_id not found
            extra = {"run_id": run_id}
            etl_logger.warning(f"No statistics found for ETL run {run_id}", extra=extra)

    # Update summary file
    log_dir = get_etl_log_dir()
    summary_file = log_dir / "etl_runs_summary.json"

    try:
        with open(summary_file, "r") as f:
            summary = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        summary = []

    # Update run in summary
    for run in summary:
        if run["run_id"] == run_id:
            run["end_time"] = datetime.now().isoformat()
            run["status"] = status
            if error:
                run["error"] = error
            break

    # Write updated summary
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


def log_company_processing(
    run_id: str,
    ticker: str,
    status: str,
    filings_processed: int = 0,
    filings_failed: int = 0,
    filings_skipped: int = 0,
    error: Optional[str] = None,
) -> None:
    """Log company processing results.

    Args:
        run_id: Unique identifier for this ETL run
        ticker: Company ticker symbol
        status: Status of company processing (completed, failed, etc.)
        filings_processed: Number of filings processed
        filings_failed: Number of filings that failed
        filings_skipped: Number of filings skipped
        error: Optional error message if processing failed
    """
    # Update run statistics
    with _run_stats_lock:
        if run_id in _run_stats:
            _run_stats[run_id]["companies_processed"] += 1
            _run_stats[run_id]["filings_processed"] += filings_processed
            _run_stats[run_id]["filings_failed"] += filings_failed
            _run_stats[run_id]["filings_skipped"] += filings_skipped
            if error:
                _run_stats[run_id]["errors"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "company": ticker,
                        "message": error,
                    }
                )

    # Log message
    extra = {"run_id": run_id}
    if status == "completed":
        etl_logger.info(
            f"Processed company {ticker}: {filings_processed} processed, "
            f"{filings_failed} failed, {filings_skipped} skipped",
            extra=extra,
        )
    else:
        etl_logger.error(f"Failed to process company {ticker}: {error}", extra=extra)


def log_filing_processing(
    run_id: str,
    filing_id: str,
    company: str,
    filing_type: str,
    status: str,
    processing_time: float,
    error: Optional[str] = None,
) -> None:
    """Log filing processing results.

    Args:
        run_id: Unique identifier for this ETL run
        filing_id: Filing ID (accession number)
        company: Company ticker symbol
        filing_type: Filing type (10-K, 10-Q, etc.)
        status: Status of filing processing (completed, failed, skipped)
        processing_time: Time taken to process the filing in seconds
        error: Optional error message if processing failed
    """
    # Log message
    extra = {"run_id": run_id}
    if status == "completed":
        etl_logger.info(
            f"Processed filing {filing_id} ({company} {filing_type}) in {processing_time:.2f}s",
            extra=extra,
        )
    elif status == "skipped":
        etl_logger.info(f"Skipped filing {filing_id} ({company} {filing_type})", extra=extra)
    else:
        etl_logger.error(
            f"Failed to process filing {filing_id} ({company} {filing_type}): {error}",
            extra=extra,
        )
        # Update run statistics with error
        with _run_stats_lock:
            if run_id in _run_stats:
                _run_stats[run_id]["errors"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "filing_id": filing_id,
                        "company": company,
                        "filing_type": filing_type,
                        "message": error,
                    }
                )


def log_api_call(
    run_id: str,
    api_name: str,
    success: bool,
    response_time: float,
    error: Optional[str] = None,
) -> None:
    """Log API call results.

    Args:
        run_id: Unique identifier for this ETL run
        api_name: Name of the API called
        success: Whether the API call was successful
        response_time: Time taken for the API call in seconds
        error: Optional error message if the API call failed
    """
    # Update run statistics
    with _run_stats_lock:
        if run_id in _run_stats:
            _run_stats[run_id]["api_calls"] += 1
            if not success:
                _run_stats[run_id]["api_errors"] += 1

    # Log message
    extra = {"run_id": run_id}
    if success:
        etl_logger.debug(f"API call to {api_name} succeeded in {response_time:.2f}s", extra=extra)
    else:
        etl_logger.warning(
            f"API call to {api_name} failed in {response_time:.2f}s: {error}",
            extra=extra,
        )


def log_rate_limit_adjustment(run_id: str, old_rate: float, new_rate: float, reason: str) -> None:
    """Log rate limit adjustment.

    Args:
        run_id: Unique identifier for this ETL run
        old_rate: Previous rate limit in seconds
        new_rate: New rate limit in seconds
        reason: Reason for the adjustment
    """
    # Update run statistics
    with _run_stats_lock:
        if run_id in _run_stats:
            _run_stats[run_id]["rate_limit_history"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "old_rate": old_rate,
                    "new_rate": new_rate,
                    "reason": reason,
                }
            )

    # Log message
    extra = {"run_id": run_id}
    etl_logger.info(
        f"Rate limit adjusted from {old_rate:.2f}s to {new_rate:.2f}s: {reason}",
        extra=extra,
    )


def log_embedding_stats(run_id: str, tokens_used: int, chunks_processed: int, fallback_count: int) -> None:
    """Log embedding generation statistics.

    Args:
        run_id: Unique identifier for this ETL run
        tokens_used: Number of tokens used
        chunks_processed: Number of chunks processed
        fallback_count: Number of fallbacks used
    """
    # Update run statistics
    with _run_stats_lock:
        if run_id in _run_stats:
            _run_stats[run_id]["embedding_stats"]["total_tokens"] += tokens_used
            _run_stats[run_id]["embedding_stats"]["total_chunks"] += chunks_processed
            _run_stats[run_id]["embedding_stats"]["fallback_count"] += fallback_count

    # Log message
    extra = {"run_id": run_id}
    etl_logger.info(
        f"Embedding stats: {tokens_used} tokens, {chunks_processed} chunks, {fallback_count} fallbacks",
        extra=extra,
    )


@contextmanager
def log_phase_timing(run_id: str, phase_name: str):
    """Context manager to log timing for a specific phase of the ETL process.

    Args:
        run_id: Unique identifier for this ETL run
        phase_name: Name of the phase being timed
    """
    start_time = time.time()
    extra = {"run_id": run_id}
    etl_logger.info(f"Starting phase: {phase_name}", extra=extra)

    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time

        # Update run statistics
        with _run_stats_lock:
            if run_id in _run_stats:
                if "phase_timings" not in _run_stats[run_id]:
                    _run_stats[run_id]["phase_timings"] = {}

                if phase_name not in _run_stats[run_id]["phase_timings"]:
                    _run_stats[run_id]["phase_timings"][phase_name] = []

                _run_stats[run_id]["phase_timings"][phase_name].append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                    }
                )

        etl_logger.info(f"Completed phase: {phase_name} in {duration:.2f}s", extra=extra)


def get_etl_run_stats(run_id: str) -> Dict[str, Any]:
    """Get statistics for a specific ETL run.

    Args:
        run_id: Unique identifier for the ETL run

    Returns:
        Dictionary of run statistics
    """
    with _run_stats_lock:
        if run_id in _run_stats:
            return _run_stats[run_id].copy()
        else:
            # Try to load from file
            stats_file = get_etl_log_dir() / f"{run_id}_stats.json"
            if stats_file.exists():
                try:
                    with open(stats_file, "r") as f:
                        return json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    return {}
            return {}


def get_recent_etl_runs(limit: int = 10) -> List[Dict[str, Any]]:
    """Get a list of recent ETL runs.

    Args:
        limit: Maximum number of runs to return

    Returns:
        List of run summaries
    """
    log_dir = get_etl_log_dir()
    summary_file = log_dir / "etl_runs_summary.json"

    try:
        with open(summary_file, "r") as f:
            summary = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

    # Sort by start time (most recent first)
    summary.sort(key=lambda x: x.get("start_time", ""), reverse=True)

    # Return limited number of runs
    return summary[:limit]


def get_etl_run_logs(run_id: str) -> List[str]:
    """Get logs for a specific ETL run.

    Args:
        run_id: Unique identifier for the ETL run

    Returns:
        List of log lines for the run
    """
    log_dir = get_etl_log_dir()

    # Find the log file for the run date
    run_date = run_id.split("_")[1][:8]  # Extract YYYYMMDD from run_id
    log_file = log_dir / f"etl_{run_date}.log"

    if not log_file.exists():
        return []

    # Extract logs for this run_id
    logs = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                if f"[{run_id}]" in line:
                    logs.append(line.strip())
    except Exception:
        return []

    return logs


def generate_etl_report(run_id: str) -> str:
    """Generate a detailed report for an ETL run.

    Args:
        run_id: Unique identifier for the ETL run

    Returns:
        Formatted report string
    """
    stats = get_etl_run_stats(run_id)
    if not stats:
        return f"No statistics found for ETL run {run_id}"

    # Calculate duration
    start_time = stats.get("start_time", 0)
    end_time = stats.get("end_time", time.time())
    duration = end_time - start_time

    # Format the report
    report = f"ETL Run Report: {run_id}\n"
    report += "=" * 50 + "\n\n"

    # Basic information
    report += f"Status: {stats.get('status', 'unknown')}\n"
    report += f"Duration: {duration:.2f} seconds\n"
    report += f"Description: {stats.get('description', 'N/A')}\n"
    report += f"Database: {stats.get('database_path', 'N/A')}\n\n"

    # Parameters
    report += "Parameters:\n"
    for key, value in stats.get("parameters", {}).items():
        report += f"  {key}: {value}\n"
    report += "\n"

    # Processing statistics
    report += "Processing Statistics:\n"
    report += f"  Companies Processed: {stats.get('companies_processed', 0)}\n"
    report += f"  Filings Processed: {stats.get('filings_processed', 0)}\n"
    report += f"  Filings Failed: {stats.get('filings_failed', 0)}\n"
    report += f"  Filings Skipped: {stats.get('filings_skipped', 0)}\n"
    report += f"  API Calls: {stats.get('api_calls', 0)}\n"
    report += f"  API Errors: {stats.get('api_errors', 0)}\n\n"

    # Embedding statistics
    embedding_stats = stats.get("embedding_stats", {})
    report += "Embedding Statistics:\n"
    report += f"  Total Tokens: {embedding_stats.get('total_tokens', 0)}\n"
    report += f"  Total Chunks: {embedding_stats.get('total_chunks', 0)}\n"
    report += f"  Fallback Count: {embedding_stats.get('fallback_count', 0)}\n\n"

    # Phase timings
    phase_timings = stats.get("phase_timings", {})
    if phase_timings:
        report += "Phase Timings:\n"
        for phase, timings in phase_timings.items():
            total_duration = sum(t.get("duration", 0) for t in timings)
            avg_duration = total_duration / len(timings) if timings else 0
            report += f"  {phase}: {total_duration:.2f}s total, {avg_duration:.2f}s avg\n"
        report += "\n"

    # Rate limit history
    rate_limit_history = stats.get("rate_limit_history", [])
    if rate_limit_history:
        report += "Rate Limit Adjustments:\n"
        for adjustment in rate_limit_history[-5:]:  # Show last 5 adjustments
            report += f"  {adjustment.get('timestamp', '')}: "
            report += f"{adjustment.get('old_rate', 0):.2f}s → {adjustment.get('new_rate', 0):.2f}s "
            report += f"({adjustment.get('reason', '')})\n"
        if len(rate_limit_history) > 5:
            report += f"  ... and {len(rate_limit_history) - 5} more adjustments\n"
        report += "\n"

    # Errors
    errors = stats.get("errors", [])
    if errors:
        report += "Errors:\n"
        for error in errors:
            report += f"  {error.get('timestamp', '')}: "
            if "company" in error:
                report += f"{error.get('company', '')} - "
            if "filing_id" in error:
                report += f"{error.get('filing_id', '')} - "
            report += f"{error.get('message', '')}\n"
        report += "\n"

    return report
