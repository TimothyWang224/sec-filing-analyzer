"""
Consolidate dependencies from requirements.txt files into pyproject.toml.

This script:
1. Parses requirements.txt and requirements-tools.txt
2. Identifies dependencies not already in pyproject.toml
3. Adds them to the appropriate sections in pyproject.toml
4. Provides a summary of changes
"""

import re
import tomlkit
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def parse_requirements(file_path):
    """Parse a requirements.txt file and return a dictionary of package names and versions."""
    requirements = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse package name and version constraint
                match = re.match(r'^([a-zA-Z0-9_\-\.]+)([<>=!~].+)?$', line)
                if match:
                    package_name, version_constraint = match.groups()
                    requirements[package_name] = version_constraint or ""
    except FileNotFoundError:
        console.print(f"[yellow]Warning: {file_path} not found[/yellow]")
    
    return requirements

def parse_pyproject_toml(file_path):
    """Parse pyproject.toml and return the document and a set of existing dependencies."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    doc = tomlkit.parse(content)
    existing_deps = set()
    
    # Check project.dependencies
    if 'project' in doc and 'dependencies' in doc['project']:
        for dep in doc['project']['dependencies']:
            if isinstance(dep, str):
                package_name = re.match(r'^([a-zA-Z0-9_\-\.]+)', dep).group(1)
                existing_deps.add(package_name)
    
    # Check tool.poetry.dependencies
    if 'tool' in doc and 'poetry' in doc['tool'] and 'dependencies' in doc['tool']['poetry']:
        for dep_name in doc['tool']['poetry']['dependencies']:
            if dep_name != 'python':
                existing_deps.add(dep_name)
    
    # Check dev dependencies
    if 'tool' in doc and 'poetry' in doc['tool'] and 'group' in doc['tool']['poetry'] and 'dev' in doc['tool']['poetry']['group'] and 'dependencies' in doc['tool']['poetry']['group']['dev']:
        for dep_name in doc['tool']['poetry']['group']['dev']['dependencies']:
            existing_deps.add(dep_name)
    
    return doc, existing_deps

def add_dependencies_to_pyproject(doc, new_deps, group=None):
    """Add new dependencies to pyproject.toml document."""
    added_deps = []
    
    if group:
        # Add to a specific group
        if 'tool' not in doc:
            doc['tool'] = tomlkit.table()
        
        if 'poetry' not in doc['tool']:
            doc['tool']['poetry'] = tomlkit.table()
        
        if 'group' not in doc['tool']['poetry']:
            doc['tool']['poetry']['group'] = tomlkit.table()
        
        if group not in doc['tool']['poetry']['group']:
            doc['tool']['poetry']['group'][group] = tomlkit.table()
        
        if 'dependencies' not in doc['tool']['poetry']['group'][group]:
            doc['tool']['poetry']['group'][group]['dependencies'] = tomlkit.table()
        
        for name, version in new_deps.items():
            # Convert pip-style version constraints to Poetry-style
            if version:
                if version.startswith('>='):
                    poetry_version = '^' + version[2:]
                else:
                    poetry_version = version
            else:
                poetry_version = '*'
            
            doc['tool']['poetry']['group'][group]['dependencies'][name] = poetry_version
            added_deps.append((name, poetry_version, group))
    else:
        # Add to main dependencies
        if 'tool' not in doc:
            doc['tool'] = tomlkit.table()
        
        if 'poetry' not in doc['tool']:
            doc['tool']['poetry'] = tomlkit.table()
        
        if 'dependencies' not in doc['tool']['poetry']:
            doc['tool']['poetry']['dependencies'] = tomlkit.table()
        
        for name, version in new_deps.items():
            # Convert pip-style version constraints to Poetry-style
            if version:
                if version.startswith('>='):
                    poetry_version = '^' + version[2:]
                else:
                    poetry_version = version
            else:
                poetry_version = '*'
            
            doc['tool']['poetry']['dependencies'][name] = poetry_version
            added_deps.append((name, poetry_version, 'main'))
    
    return added_deps

def main():
    console.print("[bold]Consolidating dependencies into pyproject.toml...[/bold]")
    
    # Parse requirements files
    req_deps = parse_requirements('requirements.txt')
    tools_deps = parse_requirements('requirements-tools.txt')
    
    console.print(f"Found [green]{len(req_deps)}[/green] dependencies in requirements.txt")
    console.print(f"Found [green]{len(tools_deps)}[/green] dependencies in requirements-tools.txt")
    
    # Parse pyproject.toml
    pyproject_path = 'pyproject.toml'
    doc, existing_deps = parse_pyproject_toml(pyproject_path)
    
    console.print(f"Found [green]{len(existing_deps)}[/green] existing dependencies in pyproject.toml")
    
    # Identify new dependencies
    new_main_deps = {name: version for name, version in req_deps.items() if name not in existing_deps}
    new_tools_deps = {name: version for name, version in tools_deps.items() if name not in existing_deps and name not in new_main_deps}
    
    # Add new dependencies to pyproject.toml
    added_main = add_dependencies_to_pyproject(doc, new_main_deps)
    added_tools = add_dependencies_to_pyproject(doc, new_tools_deps, group='tools')
    
    # Create backup of original pyproject.toml
    backup_path = pyproject_path + '.bak'
    Path(pyproject_path).rename(backup_path)
    console.print(f"Created backup of original pyproject.toml at [blue]{backup_path}[/blue]")
    
    # Write updated pyproject.toml
    with open(pyproject_path, 'w') as f:
        f.write(tomlkit.dumps(doc))
    
    # Print summary
    console.print("\n[bold green]Dependencies consolidated successfully![/bold green]")
    
    if added_main or added_tools:
        table = Table(title="Added Dependencies")
        table.add_column("Package", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Group", style="yellow")
        
        for name, version, group in added_main + added_tools:
            table.add_row(name, version, group)
        
        console.print(table)
        
        console.print("\n[bold]Next steps:[/bold]")
        console.print("1. Run [green]poetry lock[/green] to update the lock file")
        console.print("2. Run [green]poetry install[/green] to install the new dependencies")
        console.print("3. Test your project to ensure everything works correctly")
        console.print("4. If everything works, you can safely remove requirements.txt and requirements-tools.txt")
    else:
        console.print("No new dependencies were added. All dependencies are already in pyproject.toml.")

if __name__ == "__main__":
    main()
