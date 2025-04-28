#!/usr/bin/env python
"""
Simplified ETLConfig for the demo.

This module provides a simplified version of the ETLConfig class for the demo.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ETLConfig:
    """Configuration for the ETL pipeline."""

    # Data retrieval settings
    filings_dir: Path = Path("data/filings")
    filing_types: List[str] = field(default_factory=lambda: ["10-K", "10-Q", "8-K"])
    max_retries: int = 2
    timeout: int = 30

    # Document processing settings
    chunk_size: int = 1024
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"

    # Parallel processing settings
    use_parallel: bool = True
    max_workers: int = 4
    batch_size: int = 100
    rate_limit: float = 0.1

    # XBRL extraction settings
    process_quantitative: bool = True
    db_path: str = "data/db_backup/financial_data.duckdb"
    db_read_only: bool = True  # Default to read-only mode for database access

    # Processing flags
    process_semantic: bool = True
    delay_between_companies: int = 1

    def __post_init__(self):
        """Initialize configuration after creation."""
        # Create filings directory and subdirectories if they don't exist
        self.filings_dir.mkdir(parents=True, exist_ok=True)
        (self.filings_dir / "raw").mkdir(exist_ok=True)
        (self.filings_dir / "html").mkdir(exist_ok=True)
        (self.filings_dir / "xml").mkdir(exist_ok=True)
        (self.filings_dir / "processed").mkdir(exist_ok=True)
        (self.filings_dir / "cache").mkdir(exist_ok=True)

    @classmethod
    def from_env(cls) -> "ETLConfig":
        """Create configuration from environment variables."""
        return cls(
            filings_dir=Path(os.getenv("SEC_FILINGS_DIR", "data/filings")),
            filing_types=os.getenv("SEC_FILING_TYPES", "10-K,10-Q,8-K").split(","),
            max_retries=int(os.getenv("SEC_MAX_RETRIES", "3")),
            timeout=int(os.getenv("SEC_TIMEOUT", "30")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            use_parallel=os.getenv("USE_PARALLEL", "true").lower() == "true",
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            batch_size=int(os.getenv("BATCH_SIZE", "100")),
            rate_limit=float(os.getenv("RATE_LIMIT", "0.1")),
            process_quantitative=os.getenv("PROCESS_QUANTITATIVE", "true").lower() == "true",
            db_path=os.getenv("DB_PATH", "data/db_backup/financial_data.duckdb"),
            db_read_only=os.getenv("DB_READ_ONLY", "true").lower() == "true",
            process_semantic=os.getenv("PROCESS_SEMANTIC", "true").lower() == "true",
            delay_between_companies=int(os.getenv("DELAY_BETWEEN_COMPANIES", "1")),
        )
