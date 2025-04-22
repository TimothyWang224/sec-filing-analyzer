# Git Hooks

This directory contains Git hooks used in the SEC Filing Analyzer project.

## Available Hooks

### post-commit

The `post-commit` hook automatically pushes changes to GitHub after each commit.

#### Installation

To install the hook, copy the appropriate file to your `.git/hooks` directory:

For Unix/Linux/macOS:
```bash
cp scripts/git/post-commit .git/hooks/
chmod +x .git/hooks/post-commit
```

For Windows:
```
copy scripts\git\post-commit.bat .git\hooks\post-commit
```

#### How It Works

1. After each commit, the hook gets the current branch name
2. It automatically pushes the changes to the GitHub remote repository
3. It displays a success or failure message

#### Troubleshooting

If the automatic push fails, you may need to:
- Check your internet connection
- Verify your GitHub credentials are properly configured
- Push manually using `git push origin <branch-name>`

## Adding New Hooks

To add new hooks to the repository:
1. Create the hook script in this directory
2. Document it in this README
3. Provide installation instructions
