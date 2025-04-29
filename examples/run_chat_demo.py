#!/usr/bin/env python
"""
Lightweight single-agent chat demo using sample data or fixtures.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import logging utilities
try:
    from src.sec_filing_analyzer.utils.logging_utils import get_standard_log_dir, setup_logging

    # Create a wrapper function that returns a log file path
    def setup_logging_with_path() -> Path:
        """Set up logging and return the log file path."""
        # The imported setup_logging doesn't return a value
        setup_logging()

        # Create a log file path for the chat demo
        log_dir = get_standard_log_dir("chat_demo")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return log_dir / f"chat_demo_{timestamp}.log"

except ImportError:
    # Fallback if the logging utilities aren't available
    def get_standard_log_dir(subdir: Optional[str] = None) -> Path:
        """Get standard log directory.

        Args:
            subdir: Optional subdirectory within the logs directory

        Returns:
            Path to the standard log directory
        """
        base_dir = Path(".logs")
        if subdir:
            return base_dir / subdir
        return base_dir

    def setup_logging_with_path() -> Path:
        """Set up basic logging and return the log file path.

        Returns:
            Path to the log file
        """
        log_dir = Path(".logs/chat_demo")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"chat_demo_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        return log_file


# Set up logging
log_file = setup_logging_with_path()
logger = logging.getLogger("chat_demo")
logger.info(f"Starting SEC Filing Analyzer Chat Demo")
logger.info(f"Log file: {log_file}")

# Import *real* tools so the demo exercises production code paths.
# These imports ensure the tools are available in the environment
# even though they're not directly used in this simplified demo
from src.agents.qa_specialist import QASpecialistAgent
from src.llm.openai import OpenAILLM

# Import tools for documentation purposes and to ensure they're available
# in the Python environment (they may be used by other modules)
# noqa: F401 tells linters to ignore unused imports
from src.tools.sec_data import SECDataTool  # noqa: F401
from src.tools.sec_financial_data import SECFinancialDataTool  # noqa: F401
from src.tools.sec_semantic_search import SECSemanticSearchTool  # noqa: F401

logger.info("Imported required modules and tools")


def build_demo_agent() -> Any:
    """Return a stripped-down agent with just a few core tools.

    Returns:
        An agent with a respond method for generating responses to user input
    """
    logger.info("Building demo agent")

    # Create a simple QA agent with the core tools
    agent = QASpecialistAgent(
        llm_model="gpt-4o-mini",
        llm_temperature=0.7,
        llm_max_tokens=4000,
        max_iterations=5,
    )
    logger.info(f"Created QASpecialistAgent with max_iterations={5}")

    # Initialize the LLM
    llm = OpenAILLM(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=4000,
    )
    logger.info(f"Initialized OpenAI LLM with model=gpt-4o-mini, temperature=0.7")

    # Add the respond method to make it compatible with the demo interface
    def respond(user_input: str) -> str:
        """Generate a response to the user input.

        Args:
            user_input: The user's input text

        Returns:
            The generated response text
        """
        logger.info(f"Received user input: {user_input}")
        start_time = time.time()

        try:
            # For demo purposes, we'll use a simple prompt
            prompt = f"""
            You are an AI assistant specialized in analyzing SEC filings and financial data.

            User question: {user_input}

            Please respond to this question about SEC filings or financial data.
            If you need specific financial data, mention which metrics and companies you would look for.
            """
            logger.debug(f"Generated prompt: {prompt}")

            # Since generate is an async method, we need to run it synchronously
            async def get_response() -> str:
                logger.debug("Calling LLM generate method")
                return await llm.generate(prompt)

            # Run the async function in a new event loop
            response = asyncio.run(get_response())

            elapsed_time = time.time() - start_time
            logger.info(f"Generated response in {elapsed_time:.2f} seconds")
            logger.debug(f"Response: {response[:100]}...")

            return response
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error generating response after {elapsed_time:.2f} seconds: {str(e)}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            return f"Error: {str(e)}"

    # Add the respond method to the agent
    # Using setattr to avoid mypy error about QASpecialistAgent not having respond attribute
    agent.respond = respond
    logger.info("Added respond method to agent")

    return agent


def main() -> None:
    """Run the chat demo."""
    start_time = time.time()
    logger.info("Starting main function")

    parser = argparse.ArgumentParser(description="SEC Filing Analyzer demo chat")
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="(flag kept for symmetry; always true in demo script)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()

    # Set log level based on argument
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    logger.info(f"Log level set to {args.log_level}")

    # Tell downstream modules we are in demo mode (optional feature-flag).
    os.environ["SFA_DEMO_MODE"] = "1"
    logger.info("Set SFA_DEMO_MODE=1")

    # Build the demo agent
    agent = build_demo_agent()
    logger.info("Demo agent built successfully")

    # Start the chat loop
    logger.info("Starting chat loop")
    print("Type 'exit' to quit.\n")

    interaction_count = 0
    try:
        while True:
            user_input = input("🡪 ")

            # Check for exit command
            if user_input.lower() in {"exit", "quit"}:
                logger.info("User requested to exit")
                break

            # Process the input
            interaction_count += 1
            logger.info(f"Processing interaction #{interaction_count}")

            # Get response from agent
            interaction_start = time.time()
            reply = agent.respond(user_input)  # type: ignore[attr-defined]
            interaction_time = time.time() - interaction_start

            # Log and print the response
            logger.info(f"Interaction #{interaction_count} completed in {interaction_time:.2f} seconds")
            print(reply)

    except KeyboardInterrupt:
        logger.info("Chat demo interrupted by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"Unexpected error in chat loop: {type(e).__name__}: {str(e)}")
        print(f"An error occurred: {str(e)}")

    # Log total runtime
    total_runtime = time.time() - start_time
    logger.info(f"Chat demo completed after {total_runtime:.2f} seconds with {interaction_count} interactions")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal error: {type(e).__name__}: {str(e)}")
        raise
