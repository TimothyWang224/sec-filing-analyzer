"""
App State Management

Simple utility for managing application state across Streamlit sessions.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Define the path for the state file
STATE_FILE = Path("data/app_state.json")


def _ensure_state_file():
    """Ensure the state file exists."""
    # Create directory if it doesn't exist
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Create file if it doesn't exist
    if not STATE_FILE.exists():
        with open(STATE_FILE, "w") as f:
            json.dump({}, f)


def get(key: str, default: Any = None) -> Any:
    """Get a value from the app state.

    Args:
        key: The key to retrieve
        default: Default value if key doesn't exist

    Returns:
        The value or default if not found
    """
    _ensure_state_file()

    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        return state.get(key, default)
    except Exception:
        return default


def set(key: str, value: Any) -> None:
    """Set a value in the app state.

    Args:
        key: The key to set
        value: The value to store
    """
    _ensure_state_file()

    try:
        # Read current state
        with open(STATE_FILE, "r") as f:
            state = json.load(f)

        # Update state
        state[key] = value

        # Write back to file
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error setting app state: {e}")


def delete(key: str) -> None:
    """Delete a key from the app state.

    Args:
        key: The key to delete
    """
    _ensure_state_file()

    try:
        # Read current state
        with open(STATE_FILE, "r") as f:
            state = json.load(f)

        # Remove key if it exists
        if key in state:
            del state[key]

        # Write back to file
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error deleting from app state: {e}")


def get_all() -> Dict[str, Any]:
    """Get the entire app state.

    Returns:
        Dictionary containing all state values
    """
    _ensure_state_file()

    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def clear() -> None:
    """Clear the entire app state."""
    _ensure_state_file()

    try:
        with open(STATE_FILE, "w") as f:
            json.dump({}, f)
    except Exception as e:
        print(f"Error clearing app state: {e}")
