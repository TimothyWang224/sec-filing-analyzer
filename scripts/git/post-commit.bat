@echo off
REM Post-commit hook to automatically push changes to GitHub after each commit

REM Get the current branch name
FOR /F "tokens=* USEBACKQ" %%F IN (`git symbolic-ref --short HEAD`) DO (
    SET BRANCH=%%F
)

echo Auto-pushing changes to GitHub (%BRANCH%)...
git push origin %BRANCH%

REM Check if push was successful
IF %ERRORLEVEL% EQU 0 (
    echo Successfully pushed changes to GitHub.
) ELSE (
    echo Failed to push changes to GitHub. You may need to push manually.
    echo Try running: git push origin %BRANCH%
)
