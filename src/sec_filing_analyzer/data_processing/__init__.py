"""
Data processing modules for SEC filing analyzer.
"""

from .edgar_xbrl_extractor import EdgarXBRLExtractor
from .edgar_xbrl_extractor_simple import EdgarXBRLExtractorSimple
from .edgar_xbrl_to_duckdb import EdgarXBRLToDuckDBExtractor
from .simplified_xbrl_extractor import SimplifiedXBRLExtractor
from .xbrl_extractor_factory import XBRLExtractorFactory

__all__ = [
    "SimplifiedXBRLExtractor",
    "EdgarXBRLExtractor",
    "EdgarXBRLExtractorSimple",
    "XBRLExtractorFactory",
    "EdgarXBRLToDuckDBExtractor",
]
