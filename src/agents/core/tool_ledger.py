"""
Tool Ledger for tracking tool calls and results.

This module provides a centralized way to track tool calls and their results,
making it easier to reference previous tool calls and build on their results.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class ToolLedger:
    """
    Ledger for tracking tool calls and their results.

    The ToolLedger maintains a chronological record of all tool calls,
    their parameters, and their results or errors. This makes it easier
    for agents to reference previous tool calls and build on their results.
    """

    def __init__(self):
        """Initialize an empty tool ledger."""
        self.entries: List[Dict[str, Any]] = []

    def record_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a tool call in the ledger.

        Args:
            tool_name: Name of the tool that was called
            args: Arguments passed to the tool
            result: Result returned by the tool (if successful)
            error: Error message (if the tool call failed)
            metadata: Additional metadata about the tool call

        Returns:
            The created ledger entry
        """
        entry = {
            "id": len(self.entries) + 1,
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "args": args,
            "status": "error" if error else "success",
            "metadata": metadata or {},
        }

        if error:
            entry["error"] = error
        else:
            entry["result"] = result

        self.entries.append(entry)
        return entry

    def get_entries(
        self,
        tool_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get entries from the ledger with optional filtering.

        Args:
            tool_name: Optional filter by tool name
            status: Optional filter by status ("success" or "error")
            limit: Maximum number of entries to return
            offset: Number of entries to skip

        Returns:
            List of matching ledger entries
        """
        filtered = self.entries

        if tool_name:
            filtered = [e for e in filtered if e["tool"] == tool_name]

        if status:
            filtered = [e for e in filtered if e["status"] == status]

        # Return the specified slice, most recent first
        return list(reversed(filtered[offset : offset + limit]))

    def get_latest_entry(self, tool_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the most recent ledger entry.

        Args:
            tool_name: Optional filter by tool name

        Returns:
            The most recent ledger entry, or None if no entries exist
        """
        entries = self.get_entries(tool_name=tool_name, limit=1)
        return entries[0] if entries else None

    def get_entry_by_id(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a ledger entry by its ID.

        Args:
            entry_id: ID of the entry to retrieve

        Returns:
            The ledger entry if found, None otherwise
        """
        for entry in self.entries:
            if entry["id"] == entry_id:
                return entry
        return None

    def clear(self) -> None:
        """Clear all entries from the ledger."""
        self.entries = []

    def to_memory_format(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Convert ledger entries to a format suitable for agent memory.

        Args:
            limit: Maximum number of entries to include

        Returns:
            List of memory items
        """
        memory_items = []

        for entry in list(reversed(self.entries))[:limit]:
            if entry["status"] == "success":
                memory_items.append(
                    {
                        "type": "tool_result",
                        "tool": entry["tool"],
                        "args": entry["args"],
                        "result": entry["result"],
                        "timestamp": entry["timestamp"],
                    }
                )
            else:
                memory_items.append(
                    {
                        "type": "tool_error",
                        "tool": entry["tool"],
                        "args": entry["args"],
                        "error": entry["error"],
                        "timestamp": entry["timestamp"],
                    }
                )

        return memory_items

    def format_for_prompt(self, limit: int = 3) -> str:
        """
        Format ledger entries for inclusion in a prompt.

        Args:
            limit: Maximum number of entries to include

        Returns:
            Formatted string with ledger entries
        """
        if not self.entries:
            return "No previous tool calls."

        formatted = "Previous Tool Calls:\n"

        for i, entry in enumerate(list(reversed(self.entries))[:limit]):
            formatted += f"\n--- Call {i + 1} ---\n"
            formatted += f"Tool: {entry['tool']}\n"
            formatted += f"Args: {json.dumps(entry['args'], indent=2)}\n"

            if entry["status"] == "success":
                # Truncate result if it's too long
                result_str = str(entry["result"])
                if len(result_str) > 500:
                    result_str = result_str[:497] + "..."
                formatted += f"Result: {result_str}\n"
            else:
                formatted += f"Error: {entry['error']}\n"

        return formatted
