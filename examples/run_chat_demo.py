#!/usr/bin/env python
"""
Lightweight single-agent chat demo using sample data or fixtures.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import *real* tools so the demo exercises production code paths.
# Import the agent base class
from src.agents.qa_specialist import QASpecialistAgent
from src.llm.openai import OpenAILLM
from src.tools.sec_data import SECDataTool
from src.tools.sec_financial_data import SECFinancialDataTool
from src.tools.sec_semantic_search import SECSemanticSearchTool


def build_demo_agent():
    """Return a stripped-down agent with just a few core tools."""
    # Create a simple QA agent with the core tools
    agent = QASpecialistAgent(
        llm_model="gpt-4o-mini",
        llm_temperature=0.7,
        llm_max_tokens=4000,
        max_iterations=5,
    )

    # Initialize the LLM
    llm = OpenAILLM(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=4000,
    )

    # Add the respond method to make it compatible with the demo interface
    def respond(user_input):
        """Generate a response to the user input."""
        try:
            # For demo purposes, we'll use a simple prompt
            prompt = f"""
            You are an AI assistant specialized in analyzing SEC filings and financial data.

            User question: {user_input}

            Please respond to this question about SEC filings or financial data.
            If you need specific financial data, mention which metrics and companies you would look for.
            """

            # Since generate is an async method, we need to run it synchronously
            import asyncio

            async def get_response():
                return await llm.generate(prompt)

            # Run the async function in a new event loop
            response = asyncio.run(get_response())
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    # Add the respond method to the agent
    agent.respond = respond

    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="SEC Filing Analyzer demo chat")
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="(flag kept for symmetry; always true in demo script)",
    )
    _ = parser.parse_args()

    # Tell downstream modules we are in demo mode (optional feature-flag).
    os.environ["SFA_DEMO_MODE"] = "1"

    agent = build_demo_agent()

    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("🡪 ")
        if user_input.lower() in {"exit", "quit"}:
            break
        reply = agent.respond(user_input)
        print(reply)


if __name__ == "__main__":
    main()
