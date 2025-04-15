"""
LLM module for SEC Filing Analyzer.

This module provides interfaces and implementations for large language models.
"""

from .base import LLM
from .openai import OpenAILLM

__all__ = [
    'LLM',
    'OpenAILLM'
]
