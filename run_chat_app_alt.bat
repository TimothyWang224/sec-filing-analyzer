@echo off
echo Starting SEC Filing Analyzer Chat App (Alternative Method)...

:: Check if Poetry is installed
where poetry >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Poetry not found. Please install Poetry: https://python-poetry.org/docs/#installation
    pause
    exit /b 1
)

:: Install dependencies using Poetry
echo Installing dependencies using Poetry...
poetry install

:: Run the chat app using Poetry
echo Launching chat app...
poetry run python run_chat_app_alt.py
pause
