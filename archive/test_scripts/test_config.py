"""
Simple test script to verify that the configuration can be imported.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from src.sec_filing_analyzer.config import AGENT_CONFIG

    print("Successfully imported AGENT_CONFIG:")
    print(AGENT_CONFIG)
except Exception as e:
    print(f"Error importing AGENT_CONFIG: {str(e)}")
    import traceback

    traceback.print_exc()

try:
    from src.sec_filing_analyzer.llm import get_agent_config

    print("\nSuccessfully imported get_agent_config")

    # Try to get a config
    try:
        config = get_agent_config("coordinator")
        print("Successfully got coordinator config:")
        print(config)
    except Exception as e:
        print(f"Error getting coordinator config: {str(e)}")
        import traceback

        traceback.print_exc()
except Exception as e:
    print(f"Error importing get_agent_config: {str(e)}")
    import traceback

    traceback.print_exc()
