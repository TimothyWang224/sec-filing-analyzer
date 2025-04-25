"""
Time Awareness Capability

This module provides a capability for agents to understand and reason about
temporal aspects of financial data, such as fiscal periods, filing dates,
and time series analysis.
"""

import calendar
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..agents.base import Agent
from .base import Capability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeAwarenessCapability(Capability):
    """Capability for understanding and reasoning about temporal aspects of financial data."""

    def __init__(self):
        """Initialize the time awareness capability."""
        super().__init__(
            name="time_awareness",
            description="Enables agents to understand and reason about temporal aspects of financial data",
        )
        self.current_time = datetime.now()
        self.fiscal_periods = {}
        self.time_series_data = {}

    async def init(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the capability with agent and context.

        Args:
            agent: The agent this capability belongs to
            context: Initial context for the capability

        Returns:
            Updated context
        """
        self.agent = agent
        self.context = context

        # Set current time
        self.current_time = datetime.now()

        # Initialize fiscal period knowledge
        self._initialize_fiscal_periods()

        # Add time awareness to context
        context["time_awareness"] = {
            "current_time": self.current_time.isoformat(),
            "current_year": self.current_time.year,
            "current_quarter": self._get_current_quarter(),
            "fiscal_periods": self.fiscal_periods,
        }

        return context

    async def process_prompt(self, agent: Agent, context: Dict[str, Any], prompt: str) -> str:
        """
        Process the prompt to enhance time awareness.

        Args:
            agent: The agent processing the prompt
            context: Current context
            prompt: Original prompt

        Returns:
            Enhanced prompt with time awareness
        """
        # Extract temporal references from the prompt
        temporal_references = self._extract_temporal_references(prompt)

        # If temporal references are found, add time awareness context
        if temporal_references:
            time_context = self._generate_time_context(temporal_references)

            # Add time context to the prompt
            enhanced_prompt = f"{prompt}\n\nTime Context:\n"
            for key, value in time_context.items():
                enhanced_prompt += f"- {key}: {value}\n"

            return enhanced_prompt

        return prompt

    async def process_action(self, agent: Agent, context: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an action to include time awareness.

        Args:
            agent: The agent processing the action
            context: Current context
            action: Action to process

        Returns:
            Processed action with time awareness
        """
        # If the action involves querying financial data, add time parameters
        if action.get("tool") in ["sec_financial_data", "sec_semantic_search", "sec_graph_query"]:
            # Extract temporal references from the context
            temporal_references = context.get("temporal_references", {})

            # Add time parameters to the action
            if "args" not in action:
                action["args"] = {}

            # For financial data queries
            if action.get("tool") == "sec_financial_data":
                if "parameters" not in action["args"]:
                    action["args"]["parameters"] = {}

                # Add date range if available
                if "date_range" in temporal_references:
                    start_date, end_date = temporal_references["date_range"]
                    action["args"]["parameters"]["start_date"] = start_date
                    action["args"]["parameters"]["end_date"] = end_date

                # Add fiscal period if available
                if "fiscal_period" in temporal_references:
                    fiscal_year, fiscal_quarter = temporal_references["fiscal_period"]
                    action["args"]["parameters"]["fiscal_year"] = fiscal_year
                    action["args"]["parameters"]["fiscal_quarter"] = fiscal_quarter

            # For semantic search
            elif action.get("tool") == "sec_semantic_search":
                # Add date range if available
                if "date_range" in temporal_references:
                    start_date, end_date = temporal_references["date_range"]
                    action["args"]["date_range"] = (start_date, end_date)

            # For graph queries
            elif action.get("tool") == "sec_graph_query":
                if "parameters" not in action["args"]:
                    action["args"]["parameters"] = {}

                # Add fiscal year if available
                if "fiscal_period" in temporal_references:
                    fiscal_year, _ = temporal_references["fiscal_period"]
                    action["args"]["parameters"]["fiscal_year"] = fiscal_year

        return action

    async def process_result(
        self, agent: Agent, context: Dict[str, Any], response: str, action: Dict[str, Any], result: Any
    ) -> Any:
        """
        Process the result to enhance time awareness.

        Args:
            agent: The agent processing the result
            context: Current context
            response: Original response from LLM
            action: Action that produced the result
            result: Result to process

        Returns:
            Processed result with time awareness
        """
        # If the result contains time series data, store it for future reference
        if isinstance(result, dict) and "time_series" in result:
            self.time_series_data = result["time_series"]

        # If the result contains financial data, extract temporal information
        if isinstance(result, dict) and "financial_data" in result:
            financial_data = result["financial_data"]

            # Extract time periods from financial data
            time_periods = self._extract_time_periods(financial_data)

            # Add time periods to the result
            result["time_analysis"] = {
                "periods": time_periods,
                "trends": self._analyze_trends(financial_data, time_periods),
            }

        return result

    def _initialize_fiscal_periods(self) -> None:
        """Initialize knowledge about fiscal periods for common companies."""
        # Most companies follow calendar quarters
        standard_quarters = {
            "Q1": ("01-01", "03-31"),
            "Q2": ("04-01", "06-30"),
            "Q3": ("07-01", "09-30"),
            "Q4": ("10-01", "12-31"),
        }

        # Some companies have non-standard fiscal years
        # Format: {ticker: {quarter: (start_date, end_date)}}
        self.fiscal_periods = {
            "DEFAULT": standard_quarters,
            # Apple's fiscal year ends in September
            "AAPL": {
                "Q1": ("10-01", "12-31"),
                "Q2": ("01-01", "03-31"),
                "Q3": ("04-01", "06-30"),
                "Q4": ("07-01", "09-30"),
            },
            # Microsoft's fiscal year ends in June
            "MSFT": {
                "Q1": ("07-01", "09-30"),
                "Q2": ("10-01", "12-31"),
                "Q3": ("01-01", "03-31"),
                "Q4": ("04-01", "06-30"),
            },
        }

    def _get_current_quarter(self) -> str:
        """Get the current calendar quarter."""
        month = self.current_time.month
        if 1 <= month <= 3:
            return "Q1"
        elif 4 <= month <= 6:
            return "Q2"
        elif 7 <= month <= 9:
            return "Q3"
        else:
            return "Q4"

    def _extract_temporal_references(self, text: str) -> Dict[str, Any]:
        """
        Extract temporal references from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of temporal references
        """
        references = {}

        # Extract years
        year_pattern = r"\b(20\d{2})\b"
        years = re.findall(year_pattern, text)
        if years:
            references["years"] = sorted(list(set(years)))

        # Extract quarters
        quarter_pattern = r"\b(Q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\b"
        quarters = re.findall(quarter_pattern, text, re.IGNORECASE)
        if quarters:
            # Normalize quarter names
            normalized_quarters = []
            for q in quarters:
                if q.lower() == "first quarter":
                    normalized_quarters.append("Q1")
                elif q.lower() == "second quarter":
                    normalized_quarters.append("Q2")
                elif q.lower() == "third quarter":
                    normalized_quarters.append("Q3")
                elif q.lower() == "fourth quarter":
                    normalized_quarters.append("Q4")
                else:
                    normalized_quarters.append(q.upper())
            references["quarters"] = sorted(list(set(normalized_quarters)))

        # Extract date ranges
        if "years" in references:
            # If multiple years are mentioned, use them as a range
            if len(references["years"]) >= 2:
                start_year = min(references["years"])
                end_year = max(references["years"])
                references["date_range"] = (f"{start_year}-01-01", f"{end_year}-12-31")
            # If only one year is mentioned, use the whole year
            elif len(references["years"]) == 1:
                year = references["years"][0]
                references["date_range"] = (f"{year}-01-01", f"{year}-12-31")

                # If quarters are also mentioned, refine the date range
                if "quarters" in references and len(references["quarters"]) == 1:
                    quarter = references["quarters"][0]
                    if quarter == "Q1":
                        references["date_range"] = (f"{year}-01-01", f"{year}-03-31")
                    elif quarter == "Q2":
                        references["date_range"] = (f"{year}-04-01", f"{year}-06-30")
                    elif quarter == "Q3":
                        references["date_range"] = (f"{year}-07-01", f"{year}-09-30")
                    elif quarter == "Q4":
                        references["date_range"] = (f"{year}-10-01", f"{year}-12-31")

        # Extract fiscal periods
        if "years" in references and "quarters" in references:
            fiscal_year = int(references["years"][0])
            fiscal_quarter = references["quarters"][0]
            references["fiscal_period"] = (fiscal_year, fiscal_quarter)

        # Extract relative time references
        relative_pattern = r"\b(last|previous|current|next|recent)\s+(year|quarter|month|week)\b"
        relative_refs = re.findall(relative_pattern, text, re.IGNORECASE)
        if relative_refs:
            references["relative_time"] = relative_refs

            # Process relative time references
            for rel_time, rel_period in relative_refs:
                if rel_time.lower() in ["last", "previous"] and rel_period.lower() == "year":
                    last_year = self.current_time.year - 1
                    references["date_range"] = (f"{last_year}-01-01", f"{last_year}-12-31")
                elif rel_time.lower() in ["last", "previous"] and rel_period.lower() == "quarter":
                    last_quarter_date = self._get_last_quarter_date()
                    quarter_start, quarter_end = self._get_quarter_range(last_quarter_date)
                    references["date_range"] = (quarter_start, quarter_end)

        return references

    def _generate_time_context(self, temporal_references: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate time context from temporal references.

        Args:
            temporal_references: Dictionary of temporal references

        Returns:
            Dictionary of time context information
        """
        context = {}

        # Add years
        if "years" in temporal_references:
            context["Years Mentioned"] = ", ".join(temporal_references["years"])

        # Add quarters
        if "quarters" in temporal_references:
            context["Quarters Mentioned"] = ", ".join(temporal_references["quarters"])

        # Add date range
        if "date_range" in temporal_references:
            start_date, end_date = temporal_references["date_range"]
            context["Date Range"] = f"{start_date} to {end_date}"

        # Add fiscal period
        if "fiscal_period" in temporal_references:
            fiscal_year, fiscal_quarter = temporal_references["fiscal_period"]
            context["Fiscal Period"] = f"FY{fiscal_year} {fiscal_quarter}"

        # Add relative time references
        if "relative_time" in temporal_references:
            relative_refs = []
            for rel_time, rel_period in temporal_references["relative_time"]:
                relative_refs.append(f"{rel_time} {rel_period}")
            context["Relative Time References"] = ", ".join(relative_refs)

        # Add current time context
        context["Current Date"] = self.current_time.strftime("%Y-%m-%d")
        context["Current Quarter"] = self._get_current_quarter()

        return context

    def _get_last_quarter_date(self) -> datetime:
        """Get a date in the last quarter."""
        current_month = self.current_time.month
        current_year = self.current_time.year

        # Calculate last quarter's month
        if 1 <= current_month <= 3:
            last_quarter_month = 12
            last_quarter_year = current_year - 1
        else:
            last_quarter_month = ((current_month - 1) // 3) * 3
            last_quarter_year = current_year

        # Return a date in the last quarter
        return datetime(last_quarter_year, last_quarter_month, 15)

    def _get_quarter_range(self, date: datetime) -> Tuple[str, str]:
        """
        Get the date range for the quarter containing the given date.

        Args:
            date: Date to get quarter range for

        Returns:
            Tuple of (start_date, end_date) in format 'YYYY-MM-DD'
        """
        year = date.year
        month = date.month

        # Determine quarter
        if 1 <= month <= 3:
            start_month, end_month = 1, 3
        elif 4 <= month <= 6:
            start_month, end_month = 4, 6
        elif 7 <= month <= 9:
            start_month, end_month = 7, 9
        else:
            start_month, end_month = 10, 12

        # Get last day of end month
        last_day = calendar.monthrange(year, end_month)[1]

        # Format dates
        start_date = f"{year}-{start_month:02d}-01"
        end_date = f"{year}-{end_month:02d}-{last_day:02d}"

        return start_date, end_date

    def _extract_time_periods(self, financial_data: Dict[str, Any]) -> List[str]:
        """
        Extract time periods from financial data.

        Args:
            financial_data: Financial data to analyze

        Returns:
            List of time periods
        """
        time_periods = set()

        # Extract time periods from financial data
        if isinstance(financial_data, dict):
            for key, value in financial_data.items():
                if isinstance(value, dict) and "period" in value:
                    time_periods.add(value["period"])
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "period" in item:
                            time_periods.add(item["period"])

        return sorted(list(time_periods))

    def _analyze_trends(self, financial_data: Dict[str, Any], time_periods: List[str]) -> Dict[str, Any]:
        """
        Analyze trends in financial data across time periods.

        Args:
            financial_data: Financial data to analyze
            time_periods: List of time periods

        Returns:
            Dictionary of trend analysis
        """
        trends = {}

        # Simple trend analysis (placeholder)
        # In a real implementation, this would perform more sophisticated analysis
        trends["periods_count"] = len(time_periods)
        trends["has_multiple_periods"] = len(time_periods) > 1
        trends["time_span"] = f"{time_periods[0]} to {time_periods[-1]}" if len(time_periods) > 1 else time_periods[0]

        return trends
