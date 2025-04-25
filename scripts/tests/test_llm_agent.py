#!/usr/bin/env python
"""
Test script for a simple agent that uses an LLM to generate answers.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from sec_filing_analyzer.llm import OpenAILLM, get_agent_config
from src.agents.qa_specialist import QASpecialistAgent
from src.capabilities.logging import LoggingCapability
from src.capabilities.time_awareness import TimeAwarenessCapability
from src.environments.financial import FinancialEnvironment
from src.sec_filing_analyzer.utils.logging_utils import get_standard_log_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def process_question(question: str, log_level: str = "INFO", include_prompts: bool = False) -> Dict[str, Any]:
    """
    Process a question using the QA Specialist Agent with an actual LLM call.

    Args:
        question: The question to process
        log_level: Logging level
        include_prompts: Whether to include prompts and responses in logs

    Returns:
        Dictionary containing the agent's response
    """
    try:
        # Set up logging level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logging.getLogger().setLevel(numeric_level)

        logger.info(f"Processing question with LLM: {question}")

        # Initialize environment
        environment = FinancialEnvironment()

        # Get QA Specialist configuration from the master config
        llm_config = get_agent_config("qa_specialist")

        # Override with specific settings if needed
        llm_config.update(
            {
                "temperature": 0.7,  # Slightly higher for more creative responses
                "max_tokens": 1000,  # Limit response length
            }
        )

        # Initialize logging capability
        log_dir = str(get_standard_log_dir("tests"))
        os.makedirs(log_dir, exist_ok=True)

        logging_capability = LoggingCapability(
            log_dir=log_dir,
            log_level=log_level,
            log_to_console=True,
            log_to_file=True,
            include_memory=True,
            include_context=True,
            include_actions=True,
            include_results=True,
            include_prompts=include_prompts,
            include_responses=include_prompts,
            max_content_length=1000,
        )

        # Initialize time awareness capability
        time_awareness = TimeAwarenessCapability()

        # Initialize QA specialist agent with LLM config
        agent = QASpecialistAgent(
            capabilities=[logging_capability, time_awareness], environment=environment, llm_config=llm_config
        )

        # Create a custom LLM for generating the final answer
        # This is separate from the agent's internal LLM
        answer_llm = OpenAILLM(model=llm_config["model"], temperature=0.7)

        # Add the LLM to the agent for use in generating answers
        agent.answer_llm = answer_llm

        # Patch the _generate_answer method to use the LLM
        original_generate_answer = agent._generate_answer

        async def generate_answer_with_llm(input: str) -> Dict[str, Any]:
            """
            Generate an answer using the LLM.

            Args:
                input: The question to answer

            Returns:
                Dictionary containing the answer
            """
            # Call the original method to get retrieved information
            result = await original_generate_answer(input)

            # If there was an error, return the original result
            if "error" in result:
                return result

            # Extract the retrieved information
            semantic_context = result["supporting_data"]["semantic_context"]
            financial_data = result["supporting_data"]["financial_data"]
            filing_info = result["supporting_data"]["filing_info"]
            question_analysis = result["question_analysis"]

            # Create a prompt for the LLM
            prompt = f"""
            Question: {input}

            I have retrieved the following information from SEC filings:

            Semantic Context:
            {semantic_context}

            Financial Data:
            {financial_data}

            Filing Information:
            {filing_info}

            Question Analysis:
            {question_analysis}

            Based on this information, please provide a comprehensive answer to the question.
            """

            # Generate the answer using the LLM
            logger.info("Generating answer using LLM...")
            system_prompt = """You are a financial expert specializing in SEC filings analysis.
            Provide accurate, comprehensive answers based on the information provided.
            If the information is insufficient, acknowledge the limitations.
            Format your response in a clear, structured way with relevant headings and bullet points where appropriate."""

            try:
                llm_response = await agent.answer_llm.generate(
                    prompt=prompt, system_prompt=system_prompt, temperature=0.7
                )

                logger.info("LLM response received")

                # Update the result with the LLM-generated answer
                result["answer"] = llm_response
                result["explanation"] = "This answer was generated by an LLM based on retrieved SEC filing data."
                result["llm_used"] = True

                return result
            except Exception as e:
                logger.error(f"Error generating LLM response: {str(e)}")
                result["error"] = str(e)
                result["answer"] = "I encountered an error while generating the answer with the LLM."
                result["explanation"] = f"Error details: {str(e)}"
                return result

        # Replace the agent's _generate_answer method with our custom one
        agent._generate_answer = generate_answer_with_llm

        # Run the agent
        result = await agent.run(question)

        return result

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return {"error": str(e), "answer": "An error occurred while processing your question."}


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test a simple agent with LLM integration")
    parser.add_argument("--question", type=str, required=True, help="Question to ask the agent")
    parser.add_argument(
        "--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )
    parser.add_argument("--include_prompts", action="store_true", help="Include prompts and responses in logs")

    args = parser.parse_args()

    # Run the async function
    result = asyncio.run(process_question(args.question, args.log_level, args.include_prompts))

    # Print the result
    print("\n=== Agent Results ===")
    print(f"Question: {args.question}\n")
    print(f"Answer: {result.get('answer', 'No answer generated')}")


if __name__ == "__main__":
    main()
