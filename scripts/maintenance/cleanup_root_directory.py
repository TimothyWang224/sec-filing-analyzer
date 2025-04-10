"""
Clean up the root directory by moving files to appropriate directories.

This script:
1. Creates a tests/scripts directory if it doesn't exist
2. Moves test scripts to the tests/scripts directory
3. Moves utility scripts to the scripts directory
"""

import os
import shutil
from pathlib import Path
from rich.console import Console

console = Console()

def cleanup_root_directory():
    """Clean up the root directory."""
    # Define the directories
    tests_scripts_dir = Path("tests/scripts")
    scripts_dir = Path("scripts")
    docs_dir = Path("docs/assets")
    
    # Ensure the directories exist
    tests_scripts_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Ensured directory exists: {tests_scripts_dir}[/green]")
    
    # Define files to move
    test_scripts = [
        "check_neo4j.py",
        "check_neo4j_companies.py",
        "create_aapl_company.py",
        "check_vector_store.py",
        "check_vector_store_aapl.py",
        "check_vector_store_aapl_updated.py",
        "check_vector_store_simple.py",
        "search_aapl_revenue.py",
        "test_export.json"
    ]
    
    utility_scripts = [
        "migrate_data_structure.py"
    ]
    
    # Move test scripts
    for script in test_scripts:
        if os.path.exists(script):
            dest_path = tests_scripts_dir / script
            shutil.copy2(script, dest_path)
            console.print(f"[blue]Copied {script} to {tests_scripts_dir}[/blue]")
            os.remove(script)
            console.print(f"[red]Removed {script} from root directory[/red]")
        else:
            console.print(f"[yellow]File not found: {script}[/yellow]")
    
    # Move utility scripts
    for script in utility_scripts:
        if os.path.exists(script):
            dest_path = scripts_dir / script
            shutil.copy2(script, dest_path)
            console.print(f"[blue]Copied {script} to {scripts_dir}[/blue]")
            os.remove(script)
            console.print(f"[red]Removed {script} from root directory[/red]")
        else:
            console.print(f"[yellow]File not found: {script}[/yellow]")
    
    console.print("\n[bold green]Root directory cleanup complete![/bold green]")

if __name__ == "__main__":
    cleanup_root_directory()
