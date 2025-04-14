#!/usr/bin/env python
"""
Test script for an agent with dynamic termination.
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import Dict, Any, List, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.agents.qa_specialist import QASpecialistAgent
from src.agents.core.dynamic_termination import DynamicTermination
from src.environments.financial import FinancialEnvironment
from src.capabilities.logging import LoggingCapability
from src.capabilities.time_awareness import TimeAwarenessCapability
from src.sec_filing_analyzer.utils.logging_utils import get_standard_log_dir
from sec_filing_analyzer.llm import OpenAILLM, get_agent_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a QA Specialist Agent with dynamic termination
class DynamicQASpecialistAgent(QASpecialistAgent, DynamicTermination):
    """QA Specialist Agent with dynamic termination capabilities."""

    def __init__(
        self,
        capabilities: Optional[List[Any]] = None,
        max_iterations: int = 10,
        max_duration_seconds: int = 180,
        llm_config: Optional[Dict[str, Any]] = None,
        environment: Optional[FinancialEnvironment] = None,
        confidence_threshold: int = 85,
        similarity_threshold: float = 0.9,
        min_iterations: int = 1,
        enable_llm_assessment: bool = True,
        enable_convergence_check: bool = True,
        enable_confidence_check: bool = True,
        enable_info_gain_check: bool = True
    ):
        """
        Initialize the dynamic QA specialist agent.

        Args:
            capabilities: List of capabilities to extend agent behavior
            max_iterations: Maximum number of action loops
            max_duration_seconds: Maximum runtime in seconds
            llm_config: Optional LLM configuration to override defaults
            environment: Optional environment to use
            confidence_threshold: Minimum confidence level to terminate (0-100)
            similarity_threshold: Threshold for answer similarity (0.0-1.0)
            min_iterations: Minimum number of iterations before early termination
            enable_llm_assessment: Whether to use LLM self-assessment
            enable_convergence_check: Whether to check for answer convergence
            enable_confidence_check: Whether to check confidence levels
            enable_info_gain_check: Whether to check for information gain
        """
        # Initialize QA Specialist Agent
        QASpecialistAgent.__init__(
            self,
            capabilities=capabilities,
            max_iterations=max_iterations,
            max_duration_seconds=max_duration_seconds,
            llm_config=llm_config,
            environment=environment
        )

        # Initialize Dynamic Termination
        DynamicTermination.__init__(
            self,
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold,
            min_iterations=min_iterations,
            enable_llm_assessment=enable_llm_assessment,
            enable_convergence_check=enable_convergence_check,
            enable_confidence_check=enable_confidence_check,
            enable_info_gain_check=enable_info_gain_check
        )

    async def _generate_answer(self, input: str) -> Dict[str, Any]:
        """
        Generate an answer with quality assessment.

        Args:
            input: The question to answer

        Returns:
            Dictionary containing the answer and assessment
        """
        # Call the original method to get the basic answer
        result = await super()._generate_answer(input)

        # If there was an error, return the original result
        if "error" in result:
            return result

        # Assess the answer quality
        if self.enable_llm_assessment or self.enable_confidence_check:
            assessment = await self.assess_answer_quality(input, result["answer"])

            # Add assessment to result
            result["confidence"] = assessment.get("confidence", 0)
            result["should_terminate"] = assessment.get("should_terminate", "NO").upper() == "YES"
            result["assessment"] = assessment

        return result

async def process_question(
    question: str,
    log_level: str = "INFO",
    include_prompts: bool = False,
    max_iterations: int = 10,
    min_iterations: int = 1,
    confidence_threshold: int = 85,
    enable_llm_assessment: bool = True,
    enable_convergence_check: bool = True,
    enable_confidence_check: bool = True,
    enable_info_gain_check: bool = True
) -> Dict[str, Any]:
    """
    Process a question using the Dynamic QA Specialist Agent.

    Args:
        question: The question to process
        log_level: Logging level
        include_prompts: Whether to include prompts and responses in logs
        max_iterations: Maximum number of iterations
        min_iterations: Minimum number of iterations before early termination
        confidence_threshold: Minimum confidence level to terminate (0-100)
        enable_llm_assessment: Whether to use LLM self-assessment
        enable_convergence_check: Whether to check for answer convergence
        enable_confidence_check: Whether to check confidence levels
        enable_info_gain_check: Whether to check for information gain

    Returns:
        Dictionary containing the agent's response
    """
    try:
        # Set up logging level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logging.getLogger().setLevel(numeric_level)

        logger.info(f"Processing question with dynamic termination: {question}")

        # Initialize environment
        environment = FinancialEnvironment()

        # Get QA Specialist configuration from the master config
        llm_config = get_agent_config("qa_specialist")

        # Override with specific settings if needed
        llm_config.update({
            "temperature": 0.7,  # Slightly higher for more creative responses
            "max_tokens": 1000,  # Limit response length
        })

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
            max_content_length=1000
        )

        # Initialize time awareness capability
        time_awareness = TimeAwarenessCapability()

        # Initialize dynamic QA specialist agent
        agent = DynamicQASpecialistAgent(
            capabilities=[logging_capability, time_awareness],
            environment=environment,
            llm_config=llm_config,
            max_iterations=max_iterations,
            min_iterations=min_iterations,
            confidence_threshold=confidence_threshold,
            enable_llm_assessment=enable_llm_assessment,
            enable_convergence_check=enable_convergence_check,
            enable_confidence_check=enable_confidence_check,
            enable_info_gain_check=enable_info_gain_check
        )

        # Create a custom LLM for generating the final answer
        # This is separate from the agent's internal LLM
        answer_llm = OpenAILLM(
            model=llm_config["model"],
            temperature=0.7
        )

        # Add the LLM to the agent for use in generating answers
        agent.answer_llm = answer_llm

        # Run the agent
        result = await agent.run(question)

        return result

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return {
            "error": str(e),
            "answer": "An error occurred while processing your question."
        }

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test an agent with dynamic termination")
    parser.add_argument("--question", type=str, required=True, help="Question to ask the agent")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--include_prompts", action="store_true", help="Include prompts and responses in logs")
    parser.add_argument("--max_iterations", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--min_iterations", type=int, default=1, help="Minimum number of iterations before early termination")
    parser.add_argument("--confidence_threshold", type=int, default=85, help="Minimum confidence level to terminate (0-100)")
    parser.add_argument("--disable_llm_assessment", action="store_true", help="Disable LLM self-assessment")
    parser.add_argument("--disable_convergence_check", action="store_true", help="Disable answer convergence check")
    parser.add_argument("--disable_confidence_check", action="store_true", help="Disable confidence level check")
    parser.add_argument("--disable_info_gain_check", action="store_true", help="Disable information gain check")

    args = parser.parse_args()

    # Run the async function
    result = asyncio.run(process_question(
        args.question,
        args.log_level,
        args.include_prompts,
        args.max_iterations,
        args.min_iterations,
        args.confidence_threshold,
        not args.disable_llm_assessment,
        not args.disable_convergence_check,
        not args.disable_confidence_check,
        not args.disable_info_gain_check
    ))

    # Print the result
    print("\n=== Agent Results ===")
    print(f"Question: {args.question}\n")

    # Print termination reason if available
    if "assessment" in result.get("answer", {}):
        assessment = result["answer"]["assessment"]
        print(f"Confidence: {assessment.get('confidence', 'N/A')}")
        print(f"Termination: {'YES' if assessment.get('should_terminate') else 'NO'}")
        print(f"Reasoning: {assessment.get('reasoning', 'N/A')}\n")

    print(f"Answer: {result.get('answer', 'No answer generated')}")

    # Print iteration count
    print(f"\nCompleted in {result.get('iterations_completed', 'unknown')} iterations")

if __name__ == "__main__":
    main()
