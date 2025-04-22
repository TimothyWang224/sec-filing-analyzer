#!/bin/bash
echo "Installing Git hooks..."

# Create the git directory if it doesn't exist
mkdir -p .git/hooks

# Copy the post-commit hook
cp scripts/git/post-commit .git/hooks/
chmod +x .git/hooks/post-commit
echo "Post-commit hook installed."

echo
echo "Git hooks installed successfully."
echo "Now your changes will be automatically pushed to GitHub after each commit."
