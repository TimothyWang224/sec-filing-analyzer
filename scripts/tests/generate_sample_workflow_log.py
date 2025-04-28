#!/usr/bin/env python
"""
Generate Sample Workflow Log

A utility script to generate a sample workflow log for testing the workflow visualizer.
"""

import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SampleWorkflowLogger:
    """Sample workflow logger for testing."""

    def __init__(self, workflow_id: str, log_dir: Path):
        """Initialize the logger."""
        self.workflow_id = workflow_id
        self.log_dir = log_dir
        self.log_file = log_dir / f"workflow_{workflow_id}.log"
        self.json_log_file = log_dir / f"workflow_{workflow_id}.json"

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = logging.getLogger(f"workflow.{workflow_id}")
        self.logger.setLevel(logging.INFO)

        # Add file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

        # Initialize JSON log
        with open(self.json_log_file, "w") as f:
            json.dump(
                {
                    "workflow_id": workflow_id,
                    "start_time": datetime.now().isoformat(),
                    "logs": [],
                },
                f,
                indent=2,
            )

    def log_workflow_start(self, description: str):
        """Log workflow start."""
        self.logger.info(f"Workflow started: {description}")

    def log_workflow_end(self, status: str = "completed", details: str = None):
        """Log workflow end."""
        self.logger.info(f"Workflow {status}: {details or 'No details'}")

        # Update JSON log
        with open(self.json_log_file, "r+") as f:
            data = json.load(f)
            data["end_time"] = datetime.now().isoformat()
            data["status"] = status
            if details:
                data["details"] = details
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)

    def log_step(self, step_name: str, details: str = None):
        """Log a workflow step."""
        self.logger.info(f"Step: {step_name} - {details or ''}")

    def log_timing(self, category: str, operation: str, duration: float):
        """Log timing information."""
        self.logger.info(f"TIMING: {category}:{operation} completed in {duration}s")

    def log_llm_interaction(
        self, prompt: str, response: str, prompt_tokens: int, completion_tokens: int
    ):
        """Log an LLM interaction."""
        self.logger.info(f"LLM Prompt: {prompt}")
        self.logger.info(
            f"LLM Response: tokens={prompt_tokens + completion_tokens} (prompt={prompt_tokens}, completion={completion_tokens})"
        )
        self.logger.info(f"LLM Content: {response}")

    def log_tool_call(self, tool_name: str, args: dict):
        """Log a tool call."""
        self.logger.info(f"Executing tool call 1/1: {tool_name}")
        self.logger.info(f"Tool arguments: {args}")

    def log_agent_action(self, agent_name: str, action: str, details: str = None):
        """Log an agent action."""
        agent_logger = logging.getLogger(f"agent.{agent_name}")
        agent_logger.setLevel(logging.INFO)

        # Add file handler if not already added
        if not agent_logger.handlers:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            agent_logger.addHandler(file_handler)

        agent_logger.info(f"{action}: {details or ''}")


def generate_sample_workflow_log(log_dir: Path = None, workflow_id: str = None):
    """Generate a sample workflow log."""
    # Set defaults
    if not log_dir:
        log_dir = Path("data/logs/workflows")

    if not workflow_id:
        workflow_id = f"SampleWorkflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create logger
    logger = SampleWorkflowLogger(workflow_id, log_dir)

    # Log workflow start
    logger.log_workflow_start("Sample workflow for testing visualization")

    # Define workflow steps
    steps = [
        ("Initialize", "Setting up workflow components"),
        ("Data Collection", "Collecting data from sources"),
        ("Data Processing", "Processing and transforming data"),
        ("Analysis", "Analyzing processed data"),
        ("Report Generation", "Generating final report"),
        ("Cleanup", "Cleaning up temporary resources"),
    ]

    # Define tools
    tools = [
        ("sec_data", {"ticker": "AAPL", "filing_type": "10-K"}),
        (
            "sec_semantic_search",
            {"query": "Apple financial performance", "companies": ["AAPL"]},
        ),
        (
            "sec_financial_data",
            {"query_type": "financial_facts", "parameters": {"ticker": "AAPL"}},
        ),
        (
            "sec_graph_query",
            {"query_type": "company_filings", "parameters": {"ticker": "AAPL"}},
        ),
    ]

    # Define agents
    agents = [
        "FinancialAnalystAgent.20250414_125810",
        "RiskAnalystAgent.20250414_125843",
        "QASpecialistAgent.20250414_125800",
    ]

    # Log workflow execution
    for step_name, step_details in steps:
        # Log step start
        logger.log_step(step_name, step_details)

        # Log some agent actions
        for _ in range(random.randint(1, 3)):
            agent_name = random.choice(agents)
            logger.log_agent_action(
                agent_name, "Processing", f"Processing data for {step_name}"
            )

        # Log some tool calls
        for _ in range(random.randint(1, 2)):
            tool_name, tool_args = random.choice(tools)
            logger.log_tool_call(tool_name, tool_args)

            # Log timing for tool execution
            duration = random.uniform(0.5, 3.0)
            logger.log_timing("tool", f"tool_{tool_name}", duration)

        # Log timing for step
        duration = random.uniform(1.0, 5.0)
        logger.log_timing("step", step_name.lower().replace(" ", "_"), duration)

        # Simulate some processing time
        time.sleep(0.1)

    # Log some LLM interactions
    llm_prompts = [
        "What was Apple's revenue in 2023?",
        "Analyze Apple's financial performance over the last 3 years.",
        "Identify key risks mentioned in Apple's latest 10-K filing.",
    ]

    llm_responses = [
        "Apple's revenue in 2023 was $383.29 billion, representing a 2.8% decrease from the previous year.",
        "Apple's financial performance over the last 3 years shows a mixed trend. Revenue increased from $365.82 billion in 2021 to $394.33 billion in 2022, but then decreased to $383.29 billion in 2023. Net income followed a similar pattern, rising from $94.68 billion in 2021 to $99.80 billion in 2022, then declining to $96.99 billion in 2023. Gross margin has improved steadily from 41.8% in 2021 to 44.0% in 2023, indicating better cost management despite revenue challenges.",
        "Key risks identified in Apple's latest 10-K filing include: 1) Global economic conditions and consumer spending patterns, 2) Intense competition in all business areas, 3) Supply chain disruptions and component shortages, 4) Rapid technological changes requiring continuous innovation, 5) Intellectual property challenges and litigation, 6) Regulatory pressures in multiple jurisdictions, 7) Data privacy and security concerns, 8) Dependence on manufacturing and logistics in China and other countries.",
    ]

    for i in range(3):
        # Log LLM interaction
        prompt_tokens = random.randint(200, 900)
        completion_tokens = random.randint(50, 200)
        logger.log_llm_interaction(
            prompt=llm_prompts[i],
            response=llm_responses[i],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # Log timing
        duration = random.uniform(1.0, 4.0)
        logger.log_timing("llm", "generate", duration)

    # Log workflow end
    logger.log_workflow_end("completed", "Workflow completed successfully")

    return logger.log_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate Sample Workflow Log")
    parser.add_argument("--log-dir", help="Directory to save log files")
    parser.add_argument("--workflow-id", help="Workflow ID")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else None

    try:
        log_file = generate_sample_workflow_log(log_dir, args.workflow_id)
        logger.info(f"Sample workflow log generated: {log_file}")
    except Exception as e:
        logger.error(f"Error generating sample workflow log: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
