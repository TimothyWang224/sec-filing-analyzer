#!/usr/bin/env python
"""
Migrate logs from the old location to the new standard location.

This script moves log files from the old ./logs directory to the new data/logs directory,
organizing them into appropriate subdirectories based on their content.
"""

import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.sec_filing_analyzer.utils.logging_utils import get_standard_log_dir

# Initialize console
console = Console()


def migrate_logs():
    """Migrate logs from the old location to the new standard location."""
    # Define old and new log directories
    old_log_dir = Path("logs")

    # Check if the old log directory exists
    if not old_log_dir.exists():
        console.print("[yellow]Old log directory not found. Nothing to migrate.[/yellow]")
        return

    # Get all log files from the old directory
    log_files = list(old_log_dir.glob("*.log"))
    json_files = list(old_log_dir.glob("*.json"))
    all_files = log_files + json_files

    if not all_files:
        console.print("[yellow]No log files found in the old directory. Nothing to migrate.[/yellow]")
        return

    console.print(f"[green]Found {len(all_files)} log files to migrate.[/green]")

    # Create a mapping of agent types to subdirectories
    agent_types = {
        "QASpecialistAgent": "agents",
        "FinancialAnalystAgent": "agents",
        "RiskAnalystAgent": "agents",
        "CoordinatorAgent": "agents",
        "DynamicQASpecialistAgent": "agents",
        "SimpleQAAgent": "tests",
        "UnifiedAgent": "tests",
        "LLMAgent": "tests",
    }

    # Create the new log directories
    for subdir in set(agent_types.values()):
        new_dir = get_standard_log_dir(subdir)
        new_dir.mkdir(parents=True, exist_ok=True)

    # Create a general directory for logs that don't match any agent type
    general_log_dir = get_standard_log_dir("general")
    general_log_dir.mkdir(parents=True, exist_ok=True)

    # Migrate the log files
    with Progress() as progress:
        task = progress.add_task("[cyan]Migrating log files...", total=len(all_files))

        for file in all_files:
            # Determine the agent type from the filename
            agent_type = None
            for agent in agent_types:
                if agent in file.name:
                    agent_type = agent
                    break

            # Determine the destination directory
            if agent_type:
                dest_dir = get_standard_log_dir(agent_types[agent_type])
            else:
                dest_dir = general_log_dir

            # Copy the file to the new location
            dest_file = dest_dir / file.name
            shutil.copy2(file, dest_file)

            progress.update(task, advance=1, description=f"[cyan]Migrated {file.name}")

    console.print("[green]Log migration completed successfully![/green]")
    console.print(f"[green]Migrated {len(all_files)} log files from {old_log_dir} to data/logs/[/green]")

    # Ask if the user wants to delete the old log directory
    if console.input("[yellow]Do you want to delete the old log directory? (y/n): [/yellow]").lower() == "y":
        shutil.rmtree(old_log_dir)
        console.print("[green]Old log directory deleted.[/green]")
    else:
        console.print("[yellow]Old log directory kept for reference.[/yellow]")


if __name__ == "__main__":
    console.print("[bold cyan]Log Migration Utility[/bold cyan]")
    console.print("This script will migrate log files from the old ./logs directory to the new data/logs directory.")
    console.print("The logs will be organized into appropriate subdirectories based on their content.")

    if console.input("[yellow]Do you want to proceed? (y/n): [/yellow]").lower() == "y":
        migrate_logs()
    else:
        console.print("[yellow]Migration cancelled.[/yellow]")
