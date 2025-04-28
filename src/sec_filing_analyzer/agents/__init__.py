"""
Agents for SEC Filing Analyzer.

This package provides agent implementations for the SEC Filing Analyzer.
"""

from .base import Agent, Goal
from .simple_chat import SimpleChatAgent  # re-export for demo ease

__all__ = [
    "Agent",
    "Goal",
    "SimpleChatAgent",
]
