# Test Organization

This directory contains all the tests for the SEC Filing Analyzer project. The tests are organized into the following directories:

## Directory Structure

- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Tests that verify interactions between components
- `tests/e2e/`: End-to-end tests
- `tests/tools/`: Tests for tools
- `tests/scripts/`: Tests for scripts

## Running Tests

To run all tests:

```bash
python -m pytest
```

To run a specific test file:

```bash
python -m pytest tests/unit/test_file.py
```

To run tests with verbose output:

```bash
python -m pytest -v
```

## Test Fixtures

Common test fixtures are defined in `tests/conftest.py`. These fixtures are available to all tests.

## Test Data

Test data is stored in the `tests/data` directory. This includes sample filings, embeddings, and other data used in tests.

## Test Output

Test output is stored in the `tests/output` directory. This directory is created if it doesn't exist.

## Async Tests

Async tests are supported using the pytest-asyncio plugin. The asyncio mode is set to "auto" in the pytest configuration.

## Skipping Tests

To skip a test, use the `@pytest.mark.skip` decorator:

```python
@pytest.mark.skip(reason="Not implemented yet")
def test_something():
    pass
```

## Marking Tests

To mark a test, use the `@pytest.mark.` decorator:

```python
@pytest.mark.slow
def test_something_slow():
    pass
```

Then run only those tests:

```bash
python -m pytest -m slow
```

Or exclude those tests:

```bash
python -m pytest -m "not slow"
```
