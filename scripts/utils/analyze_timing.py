#!/usr/bin/env python
"""
Analyze Timing Information in Logs

This script analyzes timing information in log files to identify bottlenecks
and performance issues in the SEC Filing Analyzer.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.sec_filing_analyzer.utils.logging_utils import get_standard_log_dir

console = Console()


def parse_timing_logs(log_file: Path) -> List[Dict[str, Any]]:
    """
    Parse timing information from a log file.

    Args:
        log_file: Path to the log file

    Returns:
        List of timing entries
    """
    timing_entries = []

    # Regular expression to match timing log entries
    timing_regex = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (.*?) - (INFO|DEBUG|WARNING|ERROR) - TIMING: (.*?):(.*?) completed in (\d+\.\d+)s"
    )

    with open(log_file, "r") as f:
        for line in f:
            match = timing_regex.match(line)
            if match:
                timestamp_str, logger_name, level, category, operation, duration_str = match.groups()
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                duration = float(duration_str)

                timing_entries.append(
                    {
                        "timestamp": timestamp,
                        "logger": logger_name,
                        "level": level,
                        "category": category,
                        "operation": operation,
                        "duration": duration,
                    }
                )

    return timing_entries


def analyze_workflow_log(log_file: Path) -> Dict[str, Any]:
    """
    Analyze a workflow log file.

    Args:
        log_file: Path to the workflow log file

    Returns:
        Dictionary with analysis results
    """
    # Parse the log file
    timing_entries = parse_timing_logs(log_file)

    if not timing_entries:
        console.print(f"[yellow]No timing entries found in {log_file}[/yellow]")
        return {}

    # Group by category and operation
    categories = defaultdict(list)
    for entry in timing_entries:
        categories[entry["category"]].append(entry)

    # Calculate statistics
    stats = {}
    for category, entries in categories.items():
        operations = defaultdict(list)
        for entry in entries:
            operations[entry["operation"]].append(entry["duration"])

        operation_stats = {}
        for operation, durations in operations.items():
            operation_stats[operation] = {
                "count": len(durations),
                "total": sum(durations),
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
            }

        stats[category] = {
            "operations": operation_stats,
            "total_time": sum(entry["duration"] for entry in entries),
            "count": len(entries),
        }

    return {"file": str(log_file), "entries": timing_entries, "stats": stats}


def print_timing_summary(analysis: Dict[str, Any]):
    """
    Print a summary of timing analysis.

    Args:
        analysis: Analysis results from analyze_workflow_log
    """
    if not analysis:
        return

    console.print(f"\n[bold cyan]Timing Analysis for {analysis['file']}[/bold cyan]")

    # Create a table for each category
    for category, data in analysis["stats"].items():
        console.print(f"\n[bold green]{category.upper()} Operations[/bold green]")
        console.print(f"Total time: {data['total_time']:.2f}s, Count: {data['count']}")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Operation")
        table.add_column("Count")
        table.add_column("Total (s)")
        table.add_column("Min (s)")
        table.add_column("Max (s)")
        table.add_column("Avg (s)")
        table.add_column("% of Category")

        # Sort operations by total time (descending)
        sorted_ops = sorted(data["operations"].items(), key=lambda x: x[1]["total"], reverse=True)

        for operation, stats in sorted_ops:
            percent = (stats["total"] / data["total_time"]) * 100 if data["total_time"] > 0 else 0
            table.add_row(
                operation,
                str(stats["count"]),
                f"{stats['total']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['max']:.2f}",
                f"{stats['avg']:.2f}",
                f"{percent:.1f}%",
            )

        console.print(table)


def plot_timing_data(analysis: Dict[str, Any], output_dir: Optional[Path] = None):
    """
    Create visualizations of timing data.

    Args:
        analysis: Analysis results from analyze_workflow_log
        output_dir: Directory to save plots (if None, plots are displayed)
    """
    if not analysis or not analysis.get("stats"):
        return

    # Create output directory if needed
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Category breakdown pie chart
    category_times = {category: data["total_time"] for category, data in analysis["stats"].items()}

    plt.figure(figsize=(10, 6))
    plt.pie(category_times.values(), labels=category_times.keys(), autopct="%1.1f%%", startangle=90)
    plt.axis("equal")
    plt.title("Time Spent by Category")

    if output_dir:
        plt.savefig(output_dir / "category_breakdown.png")
        plt.close()
    else:
        plt.show()

    # 2. Top operations by time (horizontal bar chart)
    all_operations = []
    for category, data in analysis["stats"].items():
        for operation, stats in data["operations"].items():
            all_operations.append(
                {
                    "category": category,
                    "operation": operation,
                    "total_time": stats["total"],
                    "count": stats["count"],
                    "avg_time": stats["avg"],
                }
            )

    # Sort by total time and take top 15
    top_operations = sorted(all_operations, key=lambda x: x["total_time"], reverse=True)[:15]

    if top_operations:
        df = pd.DataFrame(top_operations)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(df["operation"], df["total_time"], color=[plt.cm.tab10(i % 10) for i in range(len(df))])
        plt.xlabel("Total Time (seconds)")
        plt.ylabel("Operation")
        plt.title("Top 15 Operations by Total Time")
        plt.tight_layout()

        # Add category labels to bars
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{df.iloc[i]['category']}", va="center"
            )

        if output_dir:
            plt.savefig(output_dir / "top_operations.png")
            plt.close()
        else:
            plt.show()

    # 3. Time series of operations
    if analysis.get("entries"):
        # Convert to DataFrame
        df = pd.DataFrame(analysis["entries"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        # Group by category and timestamp (1-minute bins)
        df["minute"] = df["timestamp"].dt.floor("1min")
        timeline = df.groupby(["minute", "category"])["duration"].sum().unstack().fillna(0)

        if not timeline.empty:
            plt.figure(figsize=(14, 8))
            timeline.plot.area(stacked=True, alpha=0.7, figsize=(14, 8))
            plt.xlabel("Time")
            plt.ylabel("Duration (seconds)")
            plt.title("Operation Duration Over Time by Category")
            plt.grid(True, alpha=0.3)
            plt.legend(title="Category")

            if output_dir:
                plt.savefig(output_dir / "timeline.png")
                plt.close()
            else:
                plt.show()


def find_bottlenecks(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Identify potential bottlenecks in the timing data.

    Args:
        analysis: Analysis results from analyze_workflow_log

    Returns:
        List of potential bottlenecks
    """
    bottlenecks = []

    if not analysis or not analysis.get("stats"):
        return bottlenecks

    # Look for operations that take a long time
    for category, data in analysis["stats"].items():
        for operation, stats in data["operations"].items():
            # Consider an operation a bottleneck if:
            # 1. It takes more than 1 second on average, or
            # 2. It accounts for more than 20% of its category's time
            category_percent = (stats["total"] / data["total_time"]) * 100 if data["total_time"] > 0 else 0

            if stats["avg"] > 1.0 or category_percent > 20:
                bottlenecks.append(
                    {
                        "category": category,
                        "operation": operation,
                        "avg_time": stats["avg"],
                        "total_time": stats["total"],
                        "count": stats["count"],
                        "category_percent": category_percent,
                        "severity": "high" if stats["avg"] > 3.0 or category_percent > 50 else "medium",
                    }
                )

    # Sort bottlenecks by severity and then by average time
    return sorted(bottlenecks, key=lambda x: (0 if x["severity"] == "high" else 1, -x["avg_time"]))


def print_bottlenecks(bottlenecks: List[Dict[str, Any]]):
    """
    Print identified bottlenecks.

    Args:
        bottlenecks: List of bottlenecks from find_bottlenecks
    """
    if not bottlenecks:
        console.print("\n[green]No significant bottlenecks identified.[/green]")
        return

    console.print("\n[bold red]Potential Bottlenecks[/bold red]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Category")
    table.add_column("Operation")
    table.add_column("Avg Time (s)")
    table.add_column("Total Time (s)")
    table.add_column("Count")
    table.add_column("% of Category")
    table.add_column("Severity")

    for bottleneck in bottlenecks:
        severity_color = "red" if bottleneck["severity"] == "high" else "yellow"
        table.add_row(
            bottleneck["category"],
            bottleneck["operation"],
            f"{bottleneck['avg_time']:.2f}",
            f"{bottleneck['total_time']:.2f}",
            str(bottleneck["count"]),
            f"{bottleneck['category_percent']:.1f}%",
            f"[{severity_color}]{bottleneck['severity']}[/{severity_color}]",
        )

    console.print(table)

    # Print recommendations
    console.print("\n[bold cyan]Recommendations:[/bold cyan]")
    for i, bottleneck in enumerate(bottlenecks[:3], 1):
        if bottleneck["category"] == "llm":
            console.print(
                f"[yellow]{i}. Consider using a faster LLM model for {bottleneck['operation']} operations.[/yellow]"
            )
        elif bottleneck["category"] == "tool":
            console.print(f"[yellow]{i}. Optimize the implementation of the {bottleneck['operation']} tool.[/yellow]")
        elif bottleneck["category"] == "api":
            console.print(
                f"[yellow]{i}. Consider caching results from {bottleneck['operation']} to reduce API calls.[/yellow]"
            )
        else:
            console.print(
                f"[yellow]{i}. Review the implementation of {bottleneck['operation']} in the {bottleneck['category']} category.[/yellow]"
            )


def main():
    """Main function to analyze timing logs."""
    parser = argparse.ArgumentParser(description="Analyze timing information in logs")
    parser.add_argument("--log-file", type=str, help="Path to the log file to analyze")
    parser.add_argument("--log-dir", type=str, help="Directory containing log files")
    parser.add_argument("--output-dir", type=str, help="Directory to save visualizations")
    parser.add_argument("--latest", action="store_true", help="Analyze the latest log file")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    # Determine which log file(s) to analyze
    log_files = []

    if args.log_file:
        log_files = [Path(args.log_file)]
    elif args.latest:
        # Find the latest log file
        log_dir = Path(args.log_dir) if args.log_dir else get_standard_log_dir()
        all_logs = list(log_dir.glob("**/*.log"))
        if all_logs:
            latest_log = max(all_logs, key=lambda p: p.stat().st_mtime)
            log_files = [latest_log]
            console.print(f"[green]Analyzing latest log file: {latest_log}[/green]")
        else:
            console.print("[yellow]No log files found.[/yellow]")
    elif args.log_dir:
        # Analyze all log files in the directory
        log_dir = Path(args.log_dir)
        log_files = list(log_dir.glob("*.log"))
        console.print(f"[green]Found {len(log_files)} log files in {log_dir}[/green]")
    else:
        # Default to the agents log directory
        log_dir = get_standard_log_dir("agents")
        log_files = list(log_dir.glob("*.log"))
        console.print(f"[green]Found {len(log_files)} log files in {log_dir}[/green]")

    if not log_files:
        console.print("[red]No log files to analyze. Please specify a log file or directory.[/red]")
        return

    # Analyze each log file
    for log_file in log_files:
        console.print(f"[cyan]Analyzing {log_file}...[/cyan]")
        analysis = analyze_workflow_log(log_file)

        if analysis:
            print_timing_summary(analysis)

            bottlenecks = find_bottlenecks(analysis)
            print_bottlenecks(bottlenecks)

            if not args.no_plots:
                output_dir = None
                if args.output_dir:
                    output_dir = Path(args.output_dir) / log_file.stem

                try:
                    plot_timing_data(analysis, output_dir)
                    if output_dir:
                        console.print(f"[green]Plots saved to {output_dir}[/green]")
                except Exception as e:
                    console.print(f"[red]Error generating plots: {str(e)}[/red]")
        else:
            console.print(f"[yellow]No timing data found in {log_file}[/yellow]")


if __name__ == "__main__":
    main()
