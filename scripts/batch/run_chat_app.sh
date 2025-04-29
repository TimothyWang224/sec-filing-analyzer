#!/bin/bash

echo "Starting SEC Filing Analyzer Chat App..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Please install Poetry: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Install dependencies using Poetry
echo "Installing dependencies using Poetry..."
poetry install

# Run the chat app using Poetry
echo "Launching chat app..."
poetry run python run_chat_app.py
