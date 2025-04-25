#!/usr/bin/env python
"""
Convert Workflow Log

A utility script to convert workflow logs to a structured format for visualization.
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_log_file(log_file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse a workflow log file.

    Args:
        log_file_path: Path to the log file

    Returns:
        Tuple of (log entries, workflow metadata)
    """
    log_entries = []
    workflow_metadata = {"workflow_id": "", "start_time": None, "end_time": None, "status": "unknown", "agents": set()}

    # Regular expression patterns
    timestamp_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    log_pattern = rf"{timestamp_pattern} - ([^-]+) - ([A-Z]+) - (.+)"
    timing_pattern = r"TIMING: ([^:]+):([^ ]+) completed in ([0-9.]+)s"
    tool_pattern = r"Executing tool call \d+/\d+: ([^ ]+)"
    tool_args_pattern = r"Tool arguments: (.+)"
    step_pattern = r"Step: ([^-]+) - (.*)"

    # Try to find corresponding JSON log file
    json_log_path = Path(log_file_path).with_suffix(".json")
    if json_log_path.exists():
        try:
            with open(json_log_path, "r") as f:
                json_data = json.load(f)
                workflow_metadata.update(
                    {
                        "workflow_id": json_data.get("workflow_id", ""),
                        "start_time": json_data.get("start_time"),
                        "end_time": json_data.get("end_time"),
                        "status": json_data.get("status", "unknown"),
                    }
                )
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    # Parse the log file
    with open(log_file_path, "r") as f:
        current_step = None
        current_tool = None
        current_tool_args = None

        for line in f:
            match = re.match(log_pattern, line)
            if not match:
                continue

            timestamp_str, logger_name, level, message = match.groups()
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

            # Extract workflow ID from logger name if not already set
            if not workflow_metadata["workflow_id"] and "." in logger_name:
                workflow_metadata["workflow_id"] = logger_name.split(".", 1)[1]

            # Track agents
            if "agent." in logger_name:
                agent_name = logger_name.split(".", 1)[1]
                workflow_metadata["agents"].add(agent_name)

            # Parse timing information
            timing_match = re.search(timing_pattern, message)
            if timing_match:
                category, operation, duration = timing_match.groups()
                log_entries.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "type": "timing",
                        "category": category,
                        "operation": operation,
                        "duration": float(duration),
                        "message": message,
                        "level": level,
                        "logger": logger_name,
                    }
                )
                continue

            # Parse tool execution
            tool_match = re.search(tool_pattern, message)
            if tool_match:
                current_tool = tool_match.group(1)
                continue

            tool_args_match = re.search(tool_args_pattern, message)
            if tool_args_match and current_tool:
                current_tool_args = tool_args_match.group(1)
                log_entries.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "type": "tool",
                        "tool": current_tool,
                        "args": current_tool_args,
                        "message": message,
                        "level": level,
                        "logger": logger_name,
                    }
                )
                current_tool = None
                current_tool_args = None
                continue

            # Parse workflow steps
            step_match = re.search(step_pattern, message)
            if step_match:
                step_name, step_details = step_match.groups()
                current_step = step_name.strip()
                log_entries.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "type": "step",
                        "step": current_step,
                        "details": step_details.strip(),
                        "message": message,
                        "level": level,
                        "logger": logger_name,
                    }
                )
                continue

            # Add general log entry
            log_entries.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "type": "log",
                    "message": message,
                    "level": level,
                    "logger": logger_name,
                    "current_step": current_step,
                }
            )

    # Convert agents set to list for JSON serialization
    workflow_metadata["agents"] = list(workflow_metadata["agents"])

    return log_entries, workflow_metadata


def convert_log_to_structured(log_file_path: str, output_file: Optional[str] = None) -> str:
    """
    Convert a workflow log file to a structured JSON format.

    Args:
        log_file_path: Path to the log file
        output_file: Path to save the structured JSON file (default: auto-generated)

    Returns:
        Path to the structured JSON file
    """
    log_entries, workflow_metadata = parse_log_file(log_file_path)

    # Create structured data
    structured_data = {"metadata": workflow_metadata, "entries": log_entries}

    # Determine output file path
    if not output_file:
        log_path = Path(log_file_path)
        output_file = log_path.parent / f"{log_path.stem}_structured.json"

    # Save structured data
    with open(output_file, "w") as f:
        json.dump(structured_data, f, indent=2)

    return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Convert Workflow Log")
    parser.add_argument("log_file", help="Path to workflow log file")
    parser.add_argument("--output", "-o", help="Path to save structured JSON file")
    args = parser.parse_args()

    try:
        output_file = convert_log_to_structured(args.log_file, args.output)
        logger.info(f"Structured log saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error converting log file: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
