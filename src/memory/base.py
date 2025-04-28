from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    """Represents a single item in the agent's memory."""

    content: Dict[str, Any]
    timestamp: datetime
    type: str
    metadata: Optional[Dict[str, Any]] = None


class Memory:
    """Base class for managing agent memory."""

    def __init__(self, max_items: Optional[int] = None):
        """
        Initialize memory storage.

        Args:
            max_items: Maximum number of items to store (None for unlimited)
        """
        self.items: List[MemoryItem] = []
        self.max_items = max_items

    def add(
        self,
        content: Dict[str, Any],
        type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a new item to memory.

        Args:
            content: The content to store
            type: Type of memory item (e.g., 'observation', 'action', 'result')
            metadata: Optional metadata about the memory item
        """
        item = MemoryItem(
            content=content, timestamp=datetime.now(), type=type, metadata=metadata
        )

        self.items.append(item)

        # Maintain max items limit if specified
        if self.max_items and len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]

    def get_recent(self, n: int = 5) -> List[MemoryItem]:
        """
        Get the n most recent memory items.

        Args:
            n: Number of items to retrieve

        Returns:
            List of recent memory items
        """
        return self.items[-n:]

    def get_by_type(self, type: str) -> List[MemoryItem]:
        """
        Get all memory items of a specific type.

        Args:
            type: Type of memory items to retrieve

        Returns:
            List of memory items of the specified type
        """
        return [item for item in self.items if item.type == type]

    def clear(self) -> None:
        """Clear all memory items."""
        self.items = []

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the memory contents.

        Returns:
            Dictionary containing memory statistics
        """
        return {
            "total_items": len(self.items),
            "types": {
                type: len(self.get_by_type(type))
                for type in set(item.type for item in self.items)
            },
            "oldest_item": self.items[0].timestamp if self.items else None,
            "newest_item": self.items[-1].timestamp if self.items else None,
        }
