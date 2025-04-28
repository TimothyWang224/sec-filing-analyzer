# Code Quality Tool Exclusions

This document explains the directories and files that are excluded from various code quality tools in the SEC Filing Analyzer project. These exclusions are configured in both `.pre-commit-config.yaml` and `pyproject.toml`.

## Why We Have Exclusions

Exclusions are used for several reasons:

1. **Historical code**: Some code is kept for reference but is no longer actively maintained
2. **Utility scripts**: Scripts that are used for one-off tasks or development purposes
3. **External code**: Third-party libraries or code that we don't want to modify
4. **Test code**: Test files often have different requirements than production code
5. **Generated code**: Automatically generated code that shouldn't be modified manually
6. **Special cases**: Files with unique requirements or known issues that are being addressed separately

## Current Exclusions

### Directories

| Directory | Tools | Reason |
|-----------|-------|--------|
| `archive/` | mypy, ruff, bandit | Historical code that is kept for reference but not actively maintained |
| `scripts/` | mypy, ruff, bandit | Utility scripts that don't need strict type checking or linting |
| `tests/` | bandit, ruff (partial) | Test files have different security and linting requirements |
| `chat_app/` | bandit, ruff | Chat application code with different requirements |
| `external_libs/` | bandit, ruff | Third-party libraries that we don't want to modify |
| `notebooks/` | bandit, ruff | Jupyter notebooks with different code style requirements |
| `docs/` | bandit, ruff | Documentation files, not Python code |
| `.github/` | bandit, ruff | GitHub configuration files, not Python code |
| `.logs/` | bandit, ruff | Log files, not Python code |
| `src/scripts/` | ruff (partial) | Scripts within the source directory with relaxed linting |
| `src/streamlit_app/` | ruff (partial) | Streamlit app code with relaxed linting for UI components |
| `tools/` | (previously excluded from mypy, now included) | Tool implementations that have been updated to pass type checking |

### Specific Files

| File | Tools | Reason |
|------|-------|--------|
| `src/streamlit_app/utils/__init__.py` | mypy | Has specific type issues that are being addressed separately |
| `check_company_filings.py` | bandit | Utility script with security exceptions |
| `check_db.py` | bandit | Database utility script |
| `check_financial_data.py` | bandit | Financial data utility script |
| `check_nvda_data.py` | bandit | NVIDIA data utility script |
| `debug_sec_financial_data.py` | bandit | Debugging script |
| `monitor_logs.py` | bandit | Log monitoring utility |
| `organize_root_directory.py` | bandit | Directory organization utility |
| `run_app.py` | bandit | Application runner script |
| `run_chat_app.py` | bandit | Chat application runner script |
| `run_chat_app_alt.py` | bandit | Alternative chat application runner script |
| `add_nvda.py` | bandit | NVIDIA data addition script |
| `test_*.py` | bandit | All test files |

## Disabled Rules

### mypy

The following error codes are disabled in mypy to make it more lenient:

```python
disable_error_codes = [
    "attr-defined",       # Accessing attributes that may not exist
    "no-untyped-def",     # Functions without type annotations
    "no-any-return",      # Functions returning Any
    "assignment",         # Type errors in assignments
    "arg-type",           # Type errors in function arguments
    "union-attr",         # Attribute access on union types
    "var-annotated",      # Missing variable annotations
    "list-item",          # List item type errors
    "index",              # Index access errors
    "operator",           # Operator type errors
    "call-arg",           # Function call argument errors
    "return-value",       # Return value type errors
    "name-defined",       # Undefined name errors
    "has-type",           # Missing type errors
    "override",           # Method override errors
    "return",             # Return statement errors
    "abstract",           # Abstract method errors
    "import-untyped",     # Importing modules without type stubs
    "misc",               # Miscellaneous errors
    "no-redef",           # Name redefinition errors
    "dict-item",          # Dictionary item type errors
    "call-overload"       # Function overload errors
]
```

### ruff

The following rules are ignored in ruff:

```python
ignore = [
    "E501",  # Line too long (handled by formatter)
    "S101",  # Use of assert detected (common in tests)
    "S105",  # Possible hardcoded password
    "S106",  # Possible hardcoded password
    "S108",  # Insecure usage of temp file
    "S603",  # subprocess call - check for execution of untrusted input
    "S605",  # Starting a process with a shell
    "S607",  # Starting a process with a partial executable path
    "S608",  # Possible SQL injection vector
    "S110",  # try-except-pass detected
    "S113",  # Probable use of requests call without timeout
    "S311",  # Standard pseudo-random generators are not suitable for cryptographic purposes
    "S324",  # Probable use of insecure hash functions
    "S307",  # Use of possibly insecure function
    "F401",  # Unused import
    "F403",  # Import * used
    "F811",  # Redefinition of unused name
    "F821",  # Undefined name
    "F841",  # Local variable is assigned to but never used
    "F541",  # f-string without any placeholders
    "F601",  # Dictionary key literal repeated
    "B905",  # zip() without an explicit strict parameter
    "B904",  # Within an except clause, raise exceptions with raise ... from err
    "B007",  # Loop control variable not used within loop body
    "B006",  # Do not use mutable data structures for argument defaults
    "B011",  # Do not assert False
    "B027",  # Empty method in abstract base class
    "E713",  # Test for membership should be 'not in'
    "E402",  # Module level import not at top of file
    "E722",  # Do not use bare except
    "E731",  # Do not assign a lambda expression, use a def
    "E999",  # SyntaxError
]
```

### bandit

The following security checks are skipped in bandit:

```yaml
skips:
  - B101  # Use of assert detected
  - B104  # Possible binding to all interfaces
  - B105  # Password hardcoded in config file
  - B108  # Insecure usage of temp file
  - B110  # Try, except, pass detected
  - B303  # Use of insecure MD2, MD4, MD5, or SHA1 hash function
  - B307  # Blacklisted call to eval()
  - B311  # Standard pseudo-random generators are not suitable for security/cryptographic purposes
  - B314  # Blacklisted calls to xml.etree.ElementTree
  - B324  # Use of insecure MD2, MD4, MD5, or SHA1 hash function
  - B404  # Import of subprocess module
  - B603  # subprocess call - check for execution of untrusted input
  - B605  # Starting a process with a shell
  - B607  # Starting a process with a partial executable path
  - B608  # Possible SQL injection vector
```

## Maintenance Guidelines

When working with code quality tools and exclusions:

1. **Minimize exclusions**: Try to fix issues rather than excluding files or directories
2. **Document reasons**: Always document why a file or directory is excluded
3. **Review regularly**: Periodically review exclusions to see if they can be removed
4. **Keep configurations in sync**: Ensure that exclusions are consistent between `.pre-commit-config.yaml` and `pyproject.toml`
5. **Test changes**: After modifying exclusions, run the tools to ensure they still work correctly

## Recent Changes

- The `tools/` directory was previously excluded from mypy type checking but has been updated to pass type checks and is now included
- Configurations in `pyproject.toml` and `.pre-commit-config.yaml` have been synchronized to ensure consistent behavior
