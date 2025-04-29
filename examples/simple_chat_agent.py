"""
Simple Chat Agent for SEC Filing Analyzer Demo.

This module provides a simplified chat agent for the SEC Filing Analyzer demo.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from src.agents.base import Agent
from src.agents.core import AgentState
from src.capabilities.base import Capability
from src.environments.base import Environment
from src.tools.sec_data import SECDataTool
from src.tools.sec_financial_data import SECFinancialDataTool
from src.tools.sec_semantic_search import SECSemanticSearchTool

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleChatAgent(Agent):
    """
    A simplified chat agent for the SEC Filing Analyzer demo.

    This agent uses a limited set of tools to provide a streamlined experience
    for the demo version of the SEC Filing Analyzer.
    """

    def __init__(
        self,
        environment: Optional[Environment] = None,
        capabilities: Optional[List[Capability]] = None,
        tools: Optional[List[Any]] = None,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 4000,
        max_iterations: int = 5,
        max_tool_retries: int = 2,
        tools_per_iteration: int = 1,
        vector_store_path: Optional[str] = None,
    ):
        """
        Initialize the SimpleChatAgent.

        Args:
            environment: Optional environment for the agent
            capabilities: Optional list of capabilities
            tools: Optional list of tools to use
            llm_model: LLM model to use
            llm_temperature: Temperature for LLM
            llm_max_tokens: Maximum tokens for LLM
            max_iterations: Maximum iterations for the agent
            max_tool_retries: Maximum tool retries
            tools_per_iteration: Number of tools to use per iteration
            vector_store_path: Optional path to the vector store
        """
        super().__init__(
            environment=environment,
            capabilities=capabilities or [],
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            max_iterations=max_iterations,
        )

        # Initialize state
        self.state = AgentState()
        self.state.current_iteration = 0
        self.state.start_time = time.time()
        self.state.context = {}

        # Set agent parameters
        self.max_tool_retries = max_tool_retries
        self.tools_per_iteration = tools_per_iteration

        # Initialize tools if not provided
        if tools is None:
            self.tools = [
                SECSemanticSearchTool(vector_store_path=vector_store_path),
                SECFinancialDataTool(),
                SECDataTool(),
            ]
        else:
            self.tools = tools

        # Create a mapping of tool names to tools
        self.tool_map = {tool.__class__.__name__: tool for tool in self.tools}

        # Set up the system prompt
        self.system_prompt = f"""
        You are an AI assistant specialized in analyzing SEC filings and financial data.
        You have access to the following tools:
        
        1. SECSemanticSearchTool - Use this to search for information in SEC filings
           - Best for finding qualitative information, discussions, and specific topics in filings
           - Examples: "Find NVDA's discussion of AI revenue", "Search for risk factors related to supply chain"
        
        2. SECFinancialDataTool - Use this to retrieve structured financial metrics
           - Best for quantitative financial data like revenue, profit, EPS, etc.
           - Examples: "What was Apple's revenue in 2023?", "Show me NVDA's profit margin"
        
        3. SECDataTool - Use this to get general information about companies and filings
           - Best for metadata about companies, available filings, and filing dates
           - Examples: "List available filings for AAPL", "When was NVDA's latest 10-K filed?"
        
        When answering questions:
        1. Determine which tool(s) would be most appropriate for the question
        2. Call the tool(s) with the appropriate parameters
        3. Analyze the results and provide a clear, concise answer
        4. Always cite your sources (e.g., "According to the 2023 10-K filing...")
        5. If you're unsure or don't have enough information, be honest about it
        
        You are in DEMO MODE, which means you have access to a limited set of data and functionality.
        """

    async def run(
        self,
        user_input: str,
        memory: Optional[List[Dict]] = None,
        chat_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the SimpleChatAgent with the given input.

        Args:
            user_input: The input to process
            memory: Optional memory to initialize with
            chat_mode: Whether to format the response for chat interface

        Returns:
            Dictionary containing the agent's response and any additional data
        """
        # Initialize memory if provided
        if memory:
            for item in memory:
                self.state.add_memory_item(item)

        # Initialize capabilities
        for capability in self.capabilities:
            await capability.init(self, {"input": user_input})

        # Add the user input to the context
        self.state.context["input"] = user_input

        # Reset iteration counter
        self.state.current_iteration = 0

        # Main agent loop
        while not self.should_terminate():
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, {"input": user_input}):
                    break

            # Process the input with tools
            self.state.current_iteration += 1
            logger.info(f"Starting iteration {self.state.current_iteration}")

            # Process with LLM-driven tool calling
            tool_results = await self.process_with_llm_tools(user_input)

            # Add results to memory
            self.add_to_memory(
                {
                    "type": "tool_results",
                    "content": tool_results,
                    "iteration": self.state.current_iteration,
                }
            )

            # End of loop capabilities
            for capability in self.capabilities:
                await capability.end_agent_loop(self, {"input": user_input, "results": tool_results})

        # Generate final response
        final_response = await self.generate_final_response(user_input)

        # Format response for chat mode if requested
        if chat_mode:
            return {"response": final_response}

        return final_response

    async def generate_final_response(self, user_input: str) -> str:
        """
        Generate a final response based on the memory and context.

        Args:
            user_input: The original user input

        Returns:
            The final response as a string
        """
        # Get all tool results from memory
        tool_results = []
        for item in self.state.memory:
            if item.get("type") == "tool_results":
                tool_results.append(item.get("content", {}))

        # Create a prompt for the final response
        prompt = f"""
        Based on the user's question and the tool results, provide a comprehensive answer.
        
        User question: {user_input}
        
        Tool results:
        {tool_results}
        
        Your response should:
        1. Directly answer the user's question
        2. Include relevant facts and figures from the tool results
        3. Cite sources where appropriate
        4. Be well-organized and easy to understand
        5. Acknowledge any limitations or uncertainties
        """

        # Generate the final response
        response = await self.llm.generate(prompt)

        return response
