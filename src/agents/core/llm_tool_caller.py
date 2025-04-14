"""
LLM-driven tool calling implementation.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union

from ...environments.base import Environment
from ...tools.registry import ToolRegistry
from sec_filing_analyzer.llm import BaseLLM

logger = logging.getLogger(__name__)

class LLMToolCaller:
    """
    Class for LLM-driven tool calling.
    
    This class uses an LLM to decide which tools to call and what parameters to pass,
    based on the user's question and the available tools.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        environment: Environment,
        max_tool_calls: int = 3,
        max_retries: int = 2
    ):
        """
        Initialize the LLM tool caller.
        
        Args:
            llm: LLM instance to use for tool selection
            environment: Environment containing the tools
            max_tool_calls: Maximum number of tool calls to make
            max_retries: Maximum number of retries for failed tool calls
        """
        self.llm = llm
        self.environment = environment
        self.max_tool_calls = max_tool_calls
        self.max_retries = max_retries
        
    async def select_tools(self, question: str) -> List[Dict[str, Any]]:
        """
        Use the LLM to select which tools to call and what parameters to pass.
        
        Args:
            question: User's question
            
        Returns:
            List of tool call specifications
        """
        # Get tool documentation
        tool_docs = ToolRegistry.get_tool_documentation(format="text")
        
        # Create prompt for tool selection
        prompt = self._create_tool_selection_prompt(question, tool_docs)
        
        # Generate tool selection
        system_prompt = """You are an expert at selecting the right tools to answer questions about SEC filings and financial data.
Your task is to analyze the question and select the most appropriate tools to call.
You should return a JSON array of tool calls, with each tool call specifying the tool name and parameters.
Be specific and precise with parameter values, especially company names and tickers.
If a company name is mentioned (e.g., "Apple"), include it in the appropriate parameter.
"""
        
        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2  # Low temperature for more deterministic tool selection
        )
        
        # Parse tool calls from response
        tool_calls = self._parse_tool_calls(response)
        
        return tool_calls
    
    async def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a list of tool calls.
        
        Args:
            tool_calls: List of tool call specifications
            
        Returns:
            List of tool call results
        """
        results = []
        
        for i, tool_call in enumerate(tool_calls):
            if i >= self.max_tool_calls:
                logger.warning(f"Reached maximum number of tool calls ({self.max_tool_calls}). Skipping remaining calls.")
                break
                
            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("args", {})
            
            # Execute the tool call
            for retry in range(self.max_retries + 1):
                try:
                    logger.info(f"Executing tool call {i+1}/{len(tool_calls)}: {tool_name}")
                    logger.info(f"Tool arguments: {tool_args}")
                    
                    result = await self.environment.execute_action({
                        "tool": tool_name,
                        "args": tool_args
                    })
                    
                    results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result,
                        "success": True
                    })
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    logger.error(f"Error executing tool call: {str(e)}")
                    
                    if retry < self.max_retries:
                        # Try to fix the tool call with the LLM
                        logger.info(f"Retrying tool call with LLM assistance (retry {retry+1}/{self.max_retries})")
                        fixed_args = await self._fix_tool_call(tool_name, tool_args, str(e))
                        tool_args = fixed_args
                    else:
                        # Max retries reached, add error to results
                        results.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "error": str(e),
                            "success": False
                        })
        
        return results
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a question using LLM-driven tool calling.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing tool call results and other information
        """
        # 1. Select tools to call
        tool_calls = await self.select_tools(question)
        
        # 2. Execute tool calls
        results = await self.execute_tool_calls(tool_calls)
        
        # 3. Return results
        return {
            "question": question,
            "tool_calls": tool_calls,
            "results": results
        }
    
    def _create_tool_selection_prompt(self, question: str, tool_docs: str) -> str:
        """Create a prompt for tool selection."""
        return f"""
Question: {question}

Available Tools:
{tool_docs}

Based on the question, which tool(s) should be called to retrieve the necessary information?
For each tool, specify the tool name and the parameters to pass.

Return your answer as a JSON array of tool calls, with each tool call specifying the tool name and parameters.
For example:
```json
[
  {{
    "tool": "sec_semantic_search",
    "args": {{
      "query": "Apple's revenue growth",
      "companies": ["AAPL"],
      "top_k": 5,
      "filing_types": ["10-K", "10-Q"],
      "date_range": ["2022-01-01", "2023-12-31"]
    }}
  }},
  {{
    "tool": "sec_financial_data",
    "args": {{
      "query_type": "financial_facts",
      "parameters": {{
        "ticker": "AAPL",
        "metrics": ["Revenue"],
        "start_date": "2022-01-01",
        "end_date": "2023-12-31"
      }}
    }}
  }}
]
```

Only include parameters that are relevant to the question. If a parameter is not mentioned in the question and doesn't have a default value, you can omit it.
If a company name is mentioned (e.g., "Apple"), include it in the appropriate parameter.
"""
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response."""
        # Extract JSON array from response
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find any JSON-like structure
            json_str = response
        
        try:
            # Clean up the JSON string
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            json_str = re.sub(r'[^\[\{\}\]"\':\d,\.\-\w\s]', '', json_str)
            
            # Parse the JSON
            tool_calls = json.loads(json_str)
            
            # Validate tool calls
            validated_calls = []
            for call in tool_calls:
                if isinstance(call, dict) and "tool" in call:
                    validated_calls.append(call)
            
            return validated_calls
        except Exception as e:
            logger.error(f"Error parsing tool calls: {str(e)}")
            logger.error(f"Response: {response}")
            return []
    
    async def _fix_tool_call(self, tool_name: str, tool_args: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """
        Use the LLM to fix a failed tool call.
        
        Args:
            tool_name: Name of the tool
            tool_args: Original tool arguments
            error_message: Error message from the failed call
            
        Returns:
            Fixed tool arguments
        """
        # Get tool documentation
        tool_doc = ToolRegistry.get_tool_documentation(tool_name, format="text")
        
        # Create prompt for fixing the tool call
        prompt = f"""
The following tool call failed:

Tool: {tool_name}
Arguments: {json.dumps(tool_args, indent=2)}
Error: {error_message}

Tool Documentation:
{tool_doc}

Please fix the tool arguments to make the call succeed.
Return only the fixed arguments as a JSON object.
"""
        
        system_prompt = """You are an expert at fixing failed tool calls.
Your task is to analyze the error message and tool documentation, and fix the tool arguments.
Return only the fixed arguments as a JSON object.
"""
        
        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2
        )
        
        # Parse fixed arguments from response
        try:
            # Extract JSON object from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON-like structure
                json_match = re.search(r'\{\s*".*"\s*:.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response
            
            # Clean up the JSON string
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            
            # Parse the JSON
            fixed_args = json.loads(json_str)
            
            return fixed_args
        except Exception as e:
            logger.error(f"Error parsing fixed arguments: {str(e)}")
            logger.error(f"Response: {response}")
            return tool_args  # Return original arguments if parsing fails
