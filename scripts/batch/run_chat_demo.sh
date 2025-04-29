#!/bin/bash
echo "Starting SEC Filing Analyzer Chat Demo..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Please install Poetry: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Install dependencies using Poetry
echo "Installing dependencies using Poetry..."
poetry install

# Run the chat demo using Poetry
echo "Launching chat demo..."
poetry run python examples/run_chat_demo.py
