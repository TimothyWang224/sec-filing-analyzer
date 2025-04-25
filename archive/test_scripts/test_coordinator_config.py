"""
Test script to verify that the coordinator agent is using configuration values.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.sec_filing_analyzer.config import AGENT_CONFIG
from src.sec_filing_analyzer.llm import get_agent_config


def main():
    """Test the configuration values."""
    try:
        # Get the configuration for each agent type
        financial_analyst_config = get_agent_config("financial_analyst")
        risk_analyst_config = get_agent_config("risk_analyst")
        qa_specialist_config = get_agent_config("qa_specialist")

        # Print the configuration values
        print("Financial Analyst Config:")
        print(f"  Model: {financial_analyst_config.get('model')}")
        print(f"  Temperature: {financial_analyst_config.get('temperature')}")
        print(f"  Max Tokens: {financial_analyst_config.get('max_tokens')}")

        print("\nRisk Analyst Config:")
        print(f"  Model: {risk_analyst_config.get('model')}")
        print(f"  Temperature: {risk_analyst_config.get('temperature')}")
        print(f"  Max Tokens: {risk_analyst_config.get('max_tokens')}")

        print("\nQA Specialist Config:")
        print(f"  Model: {qa_specialist_config.get('model')}")
        print(f"  Temperature: {qa_specialist_config.get('temperature')}")
        print(f"  Max Tokens: {qa_specialist_config.get('max_tokens')}")

        print("\nGlobal Agent Config:")
        print(f"  LLM Model: {AGENT_CONFIG.get('llm_model')}")
        print(f"  LLM Temperature: {AGENT_CONFIG.get('llm_temperature')}")
        print(f"  LLM Max Tokens: {AGENT_CONFIG.get('llm_max_tokens')}")

        print("\nTest successful!")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
