from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import Memory, MemoryItem


class FinancialMemory(Memory):
    """Memory implementation specialized for storing financial analysis results."""

    def __init__(self, max_items: Optional[int] = None):
        """
        Initialize the financial memory.

        Args:
            max_items: Maximum number of items to store
        """
        super().__init__(max_items)

    def add_financial_metric(
        self,
        metric_name: str,
        value: Any,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a financial metric to memory.

        Args:
            metric_name: Name of the financial metric
            value: Value of the metric
            timestamp: Optional timestamp for the metric
            metadata: Optional metadata for the metric
        """
        self.add(
            MemoryItem(
                content={
                    "type": "financial_metric",
                    "name": metric_name,
                    "value": value,
                },
                timestamp=timestamp or datetime.now(),
                type="financial_metric",
                metadata=metadata,
            )
        )

    def add_risk_assessment(
        self,
        risk_name: str,
        severity: str,
        likelihood: str,
        description: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a risk assessment to memory.

        Args:
            risk_name: Name of the risk
            severity: Severity level of the risk
            likelihood: Likelihood of the risk
            description: Description of the risk
            timestamp: Optional timestamp for the assessment
            metadata: Optional metadata for the assessment
        """
        self.add(
            MemoryItem(
                content={
                    "type": "risk_assessment",
                    "name": risk_name,
                    "severity": severity,
                    "likelihood": likelihood,
                    "description": description,
                },
                timestamp=timestamp or datetime.now(),
                type="risk_assessment",
                metadata=metadata,
            )
        )

    def add_insight(
        self,
        insight: str,
        category: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an insight to memory.

        Args:
            insight: The insight text
            category: Category of the insight
            timestamp: Optional timestamp for the insight
            metadata: Optional metadata for the insight
        """
        self.add(
            MemoryItem(
                content={"type": "insight", "text": insight, "category": category},
                timestamp=timestamp or datetime.now(),
                type="insight",
                metadata=metadata,
            )
        )

    def get_financial_metrics(self) -> List[Dict[str, Any]]:
        """
        Get all financial metrics from memory.

        Returns:
            List of financial metrics
        """
        return [item.content for item in self.get_by_type("financial_metric")]

    def get_risk_assessments(self) -> List[Dict[str, Any]]:
        """
        Get all risk assessments from memory.

        Returns:
            List of risk assessments
        """
        return [item.content for item in self.get_by_type("risk_assessment")]

    def get_insights(self) -> List[Dict[str, Any]]:
        """
        Get all insights from memory.

        Returns:
            List of insights
        """
        return [item.content for item in self.get_by_type("insight")]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the financial memory contents.

        Returns:
            Dictionary containing memory summary
        """
        summary = super().get_summary()
        summary.update(
            {
                "financial_metrics": len(self.get_financial_metrics()),
                "risk_assessments": len(self.get_risk_assessments()),
                "insights": len(self.get_insights()),
            }
        )
        return summary
