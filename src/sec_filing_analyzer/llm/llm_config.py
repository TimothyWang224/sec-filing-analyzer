from typing import Dict, Any, Optional, List
import os

# Base configuration shared across all LLMs
BASE_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Base agent execution configuration shared across all agents
BASE_AGENT_CONFIG = {
    # Agent iteration parameters
    "max_iterations": 3,  # Legacy parameter, still used for backward compatibility
    "max_planning_iterations": 2,
    "max_execution_iterations": 3,
    "max_refinement_iterations": 1,

    # Tool execution parameters
    "max_tool_retries": 2,
    "tools_per_iteration": 1,  # Default to 1 for single tool call approach
    "circuit_breaker_threshold": 3,
    "circuit_breaker_reset_timeout": 300,

    # Runtime parameters
    "max_duration_seconds": 180,

    # Termination parameters
    "enable_dynamic_termination": False,
    "min_confidence_threshold": 0.8
}

# Coordinator-specific configuration
# Uses GPT-4o for complex coordination and synthesis tasks
COORDINATOR_CONFIG = {
    # LLM parameters
    **BASE_CONFIG,
    "model": "gpt-4o",
    "temperature": 0.7,  # Balanced for creative synthesis
    "max_tokens": 2000,  # Longer responses for comprehensive reports
    "system_prompt": """You are a financial diligence coordinator. Your role is to:
1. Coordinate multiple specialized agents
2. Synthesize insights from various analyses
3. Generate comprehensive diligence reports
4. Ensure consistent and coherent analysis

Your responses should be well-structured, comprehensive, and actionable.""",

    # Agent execution parameters
    **BASE_AGENT_CONFIG,
    "max_planning_iterations": 1,
    "max_execution_iterations": 2,
    "max_refinement_iterations": 1,
    "max_duration_seconds": 300  # Longer runtime for complex coordination tasks
}

# Financial Analyst configuration
# Uses GPT-4o-mini for precise financial analysis
FINANCIAL_ANALYST_CONFIG = {
    # LLM parameters
    **BASE_CONFIG,
    "model": "gpt-4o-mini",
    "temperature": 0.3,  # Lower for precise analysis
    "max_tokens": 1000,  # Focused responses
    "system_prompt": """You are a financial analysis expert. Your responses should be:
1. Precise and data-driven
2. Focused on key financial metrics
3. Clear and concise
4. Based on standard financial analysis principles""",

    # Agent execution parameters
    **BASE_AGENT_CONFIG,
    "max_planning_iterations": 1,
    "max_execution_iterations": 2,
    "max_refinement_iterations": 1
}

# Risk Analyst configuration

# Uses GPT-4o-mini for systematic risk assessment
RISK_ANALYST_CONFIG = {
    # LLM parameters
    **BASE_CONFIG,
    "model": "gpt-4o-mini",
    "temperature": 0.3,  # Lower for precise analysis
    "max_tokens": 1000,  # Focused responses
    "system_prompt": """You are a risk analysis expert. Your responses should be:
1. Comprehensive in risk identification
2. Quantitative in risk assessment
3. Clear in risk categorization
4. Actionable in risk mitigation recommendations""",

    # Agent execution parameters
    **BASE_AGENT_CONFIG,
    "max_planning_iterations": 1,
    "max_execution_iterations": 2,
    "max_refinement_iterations": 1
}

# QA Specialist configuration
# Uses GPT-4o-mini for natural language interaction
QA_SPECIALIST_CONFIG = {
    # LLM parameters
    **BASE_CONFIG,
    "model": "gpt-4o-mini",
    "temperature": 0.5,  # Higher for natural responses
    "max_tokens": 1500,  # Longer for detailed explanations
    "system_prompt": """You are a financial Q&A expert. Your responses should be:
1. Clear and easy to understand
2. Comprehensive in explanation
3. Accurate in financial details
4. Engaging and conversational
5. Contextually aware""",

    # Agent execution parameters
    **BASE_AGENT_CONFIG,
    "max_iterations": 5,  # Increased from 3 to 5 for more complex queries
    "max_planning_iterations": 1,
    "max_execution_iterations": 5,  # Increased from 2 to 5 to allow more tool calls
    "max_refinement_iterations": 2,  # Increased from 1 to 2 for better answer refinement
    "max_tokens": 4000  # Longer responses for detailed explanations
}

# SEC Analysis capability configuration
SEC_ANALYSIS_CONFIG = {
    # LLM parameters
    **BASE_CONFIG,
    "model": "gpt-4o-mini",
    "temperature": 0.3,  # Lower for precise analysis
    "max_tokens": 1000,  # Focused responses
    "system_prompt": """You are an SEC filing analysis expert. Your responses should be:
1. Accurate in SEC filing interpretation
2. Comprehensive in data extraction
3. Clear in analysis presentation
4. Based on regulatory requirements""",

    # Agent execution parameters
    **BASE_AGENT_CONFIG,
    "max_planning_iterations": 1,
    "max_execution_iterations": 2,
    "max_refinement_iterations": 1
}

# Mapping of agent types to their configurations
AGENT_CONFIGS = {
    "coordinator": COORDINATOR_CONFIG,
    "financial_analyst": FINANCIAL_ANALYST_CONFIG,
    "risk_analyst": RISK_ANALYST_CONFIG,
    "qa_specialist": QA_SPECIALIST_CONFIG,
    "sec_analysis": SEC_ANALYSIS_CONFIG
}

def get_agent_config(agent_type: str) -> Dict[str, Any]:
    """
    Get the LLM configuration for a specific agent type.

    Args:
        agent_type: Type of agent to get configuration for

    Returns:
        Dictionary containing LLM configuration

    Raises:
        ValueError: If agent_type is not found in configurations
    """
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return AGENT_CONFIGS[agent_type].copy()  # Return a copy to prevent modification


def get_agent_types() -> List[str]:
    """
    Get all available agent types.

    Returns:
        List of agent type names
    """
    return list(AGENT_CONFIGS.keys())

class LLMConfigFactory:
    """Factory for creating and managing LLM configurations."""

    @staticmethod
    def create_config(agent_type: str, **overrides) -> Dict[str, Any]:
        """
        Create a configuration for a specific agent type.

        Args:
            agent_type: Type of agent to create configuration for
            **overrides: Configuration overrides to apply

        Returns:
            Dictionary containing the LLM configuration

        Raises:
            ValueError: If agent_type is not found in configurations
        """
        if agent_type not in AGENT_CONFIGS:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Start with base config
        config = BASE_CONFIG.copy()

        # Apply agent-specific config
        config.update(AGENT_CONFIGS[agent_type])

        # Apply any overrides
        config.update(overrides)

        return config

    @staticmethod
    def create_config_from_provider(agent_type: str, **overrides) -> Dict[str, Any]:
        """
        Create a configuration for a specific agent type using the ConfigProvider.

        Args:
            agent_type: Type of agent to create configuration for
            **overrides: Configuration overrides to apply

        Returns:
            Dictionary containing the LLM configuration

        Raises:
            ValueError: If agent_type is not found in configurations
        """
        try:
            from sec_filing_analyzer.config import ConfigProvider
            config = ConfigProvider.get_agent_config(agent_type)

            # Apply any overrides
            config.update(overrides)

            return config
        except ImportError:
            # Fall back to the old method if ConfigProvider is not available
            return LLMConfigFactory.create_config(agent_type, **overrides)

    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """
        Get available LLM models and their descriptions.

        Returns:
            Dictionary mapping model names to their descriptions
        """
        return {
            "gpt-4o": "Most capable model for complex tasks",
            "gpt-4o-mini": "Efficient model for standard tasks",
            "gpt-3.5-turbo": "Legacy model for basic tasks"
        }

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate a configuration dictionary.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        required_fields = {"model", "temperature", "max_tokens"}
        return all(field in config for field in required_fields)

    @staticmethod
    def get_recommended_config(agent_type: str, task_complexity: str = "medium") -> Dict[str, Any]:
        """
        Get a recommended configuration based on agent type and task complexity.

        Args:
            agent_type: Type of agent
            task_complexity: Complexity of the task ("low", "medium", "high")

        Returns:
            Dictionary containing recommended configuration
        """
        # Try to use the ConfigProvider first
        try:
            from sec_filing_analyzer.config import ConfigProvider
            base_config = ConfigProvider.get_agent_config(agent_type)
        except ImportError:
            # Fall back to the old method if ConfigProvider is not available
            base_config = LLMConfigFactory.create_config(agent_type)

        # Adjust configuration based on task complexity
        if task_complexity == "high":
            base_config["model"] = "gpt-4o"
            base_config["max_tokens"] = 4000
            base_config["max_planning_iterations"] = 2
            base_config["max_execution_iterations"] = 3
            base_config["max_refinement_iterations"] = 2
            base_config["max_duration_seconds"] = 300
        elif task_complexity == "low":
            base_config["model"] = "gpt-4o-mini"
            base_config["max_tokens"] = 1000
            base_config["max_planning_iterations"] = 1
            base_config["max_execution_iterations"] = 1
            base_config["max_refinement_iterations"] = 1
            base_config["max_duration_seconds"] = 120

        return base_config