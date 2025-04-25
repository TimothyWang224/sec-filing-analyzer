"""
Quantitative data processing modules for SEC filing analyzer.
"""

from .edgar_xbrl_to_duckdb import EdgarXBRLToDuckDBExtractor

__all__ = ["EdgarXBRLToDuckDBExtractor"]
