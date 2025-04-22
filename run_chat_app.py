"""
Run the SEC Filing Analyzer Chat App.

This script launches the Streamlit-based chat app for interacting with the
Financial Diligence Coordinator agent.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the SEC Filing Analyzer Chat App."""
    try:
        # Get the directory of this script
        script_dir = Path(__file__).resolve().parent

        # Set the working directory to the script directory
        os.chdir(script_dir)

        # Run the Streamlit app
        print("Starting SEC Filing Analyzer Chat App...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "chat_app/app.py"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
    except KeyboardInterrupt:
        print("Chat app stopped by user.")

if __name__ == "__main__":
    main()
