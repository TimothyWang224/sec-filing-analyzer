"""
Extract Streamlit Configuration

This script extracts the Streamlit configuration from .streamlit/config.toml
and updates the unified configuration file.
"""

import json
import os
import sys
from pathlib import Path

import toml

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    # Load the Streamlit configuration
    streamlit_config_path = Path(".streamlit/config.toml")
    if not streamlit_config_path.exists():
        print(f"Streamlit configuration file not found: {streamlit_config_path}")
        sys.exit(1)

    streamlit_config = toml.load(streamlit_config_path)
    print(f"Loaded Streamlit configuration from {streamlit_config_path}")

    # Load the unified configuration file
    config_path = Path("data/config/etl_config.json")
    if not config_path.exists():
        print(f"Unified configuration file not found: {config_path}")
        print("Creating a new configuration file...")
        config = {}
    else:
        with open(config_path, "r") as f:
            config = json.load(f)

    # Update the configuration with the Streamlit settings
    config["streamlit"] = {
        "server": {
            "port": streamlit_config.get("server", {}).get("port", 8501),
            "headless": streamlit_config.get("server", {}).get("headless", True),
            "enable_cors": streamlit_config.get("server", {}).get("enableCORS", True),
            "enable_xsrf_protection": streamlit_config.get("server", {}).get("enableXsrfProtection", False),
            "max_upload_size": streamlit_config.get("server", {}).get("maxUploadSize", 200),
            "base_url_path": streamlit_config.get("server", {}).get("baseUrlPath", ""),
            "enable_websocket_compression": streamlit_config.get("server", {}).get("enableWebsocketCompression", False),
        },
        "browser": {
            "server_address": streamlit_config.get("browser", {}).get("serverAddress", "localhost"),
            "gather_usage_stats": streamlit_config.get("browser", {}).get("gatherUsageStats", False),
            "serve_trailing_slash": streamlit_config.get("browser", {}).get("serveTrailingSlash", True),
        },
        "theme": {"base": streamlit_config.get("theme", {}).get("base", "light")},
        "ui": {"hide_top_bar": streamlit_config.get("ui", {}).get("hideTopBar", False)},
        "client": {
            "show_error_details": streamlit_config.get("client", {}).get("showErrorDetails", True),
            "toolbar_mode": streamlit_config.get("client", {}).get("toolbarMode", "auto"),
            "caching": streamlit_config.get("client", {}).get("caching", False),
            "display_enabled": streamlit_config.get("client", {}).get("displayEnabled", True),
        },
        "runner": {
            "magic_enabled": streamlit_config.get("runner", {}).get("magicEnabled", True),
            "install_tracer": streamlit_config.get("runner", {}).get("installTracer", True),
            "fix_matplotlib": streamlit_config.get("runner", {}).get("fixMatplotlib", True),
            "fast_reruns": streamlit_config.get("runner", {}).get("fastReruns", False),
        },
        "global": {
            "disable_watchdog_warning": streamlit_config.get("global", {}).get("disableWatchdogWarning", False),
            "show_warning_on_direct_execution": streamlit_config.get("global", {}).get(
                "showWarningOnDirectExecution", True
            ),
            "data_frame_serialization": streamlit_config.get("global", {}).get("dataFrameSerialization", "arrow"),
            "suppress_deprecation_warnings": streamlit_config.get("global", {}).get(
                "suppressDeprecationWarnings", True
            ),
        },
    }

    # Save the updated configuration
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated unified configuration file: {config_path}")

except Exception as e:
    print(f"Error extracting Streamlit configuration: {str(e)}")
    import traceback

    traceback.print_exc()
