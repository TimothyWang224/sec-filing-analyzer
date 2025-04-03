from typing import Dict, Any

# Base configuration shared across all LLMs
BASE_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Coordinator-specific configuration
# Uses GPT-4 for complex coordination and synthesis tasks
COORDINATOR_CONFIG = {
    **BASE_CONFIG,
    "model": "gpt-4o",
    "temperature": 0.7,  # Balanced for creative synthesis
    "max_tokens": 2000,  # Longer responses for comprehensive reports
    "system_prompt": """You are a financial diligence coordinator. Your role is to:
1. Coordinate multiple specialized agents
2. Synthesize insights from various analyses
3. Generate comprehensive diligence reports
4. Ensure consistent and coherent analysis

Your responses should be well-structured, comprehensive, and actionable."""
}

# Financial Analyst configuration
# Uses GPT-3.5 for precise financial analysis
FINANCIAL_ANALYST_CONFIG = {
    **BASE_CONFIG,
    "model": "gpt-4o-mini",
    "temperature": 0.3,  # Lower for precise analysis
    "max_tokens": 1000,  # Focused responses
    "system_prompt": """You are a financial analysis expert. Your responses should be:
1. Precise and data-driven
2. Focused on key financial metrics
3. Clear and concise
4. Based on standard financial analysis principles"""
}

# Risk Analyst configuration
# Uses GPT-3.5 for systematic risk assessment
RISK_ANALYST_CONFIG = {
    **BASE_CONFIG,
    "model": "gpt-4o-mini",
    "temperature": 0.3,  # Lower for precise analysis
    "max_tokens": 1000,  # Focused responses
    "system_prompt": """You are a risk analysis expert. Your responses should be:
1. Comprehensive in risk identification
2. Quantitative in risk assessment
3. Clear in risk categorization
4. Actionable in risk mitigation recommendations"""
}

# QA Specialist configuration
# Uses GPT-3.5 for natural language interaction
QA_SPECIALIST_CONFIG = {
    **BASE_CONFIG,
    "model": "gpt-4o-mini",
    "temperature": 0.5,  # Higher for natural responses
    "max_tokens": 1500,  # Longer for detailed explanations
    "system_prompt": """You are a financial Q&A expert. Your responses should be:
1. Clear and easy to understand
2. Comprehensive in explanation
3. Accurate in financial details
4. Engaging and conversational
5. Contextually aware"""
}

# SEC Analysis capability configuration
SEC_ANALYSIS_CONFIG = {
    **BASE_CONFIG,
    "model": "gpt-4o-mini",
    "temperature": 0.3,  # Lower for precise analysis
    "max_tokens": 1000,  # Focused responses
    "system_prompt": """You are an SEC filing analysis expert. Your responses should be:
1. Accurate in SEC filing interpretation
2. Comprehensive in data extraction
3. Clear in analysis presentation
4. Based on regulatory requirements"""
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
    def get_available_models() -> Dict[str, str]:
        """
        Get available LLM models and their descriptions.
        
        Returns:
            Dictionary mapping model names to their descriptions
        """
        return {
            "gpt-4-turbo-preview": "Most capable model for complex tasks",
            "gpt-3.5-turbo": "Efficient model for standard tasks",
            "gpt-3.5-turbo-16k": "Efficient model with larger context window"
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
        base_config = LLMConfigFactory.create_config(agent_type)
        
        # Adjust configuration based on task complexity
        if task_complexity == "high":
            base_config["model"] = "gpt-4-turbo-preview"
            base_config["max_tokens"] = 2000
        elif task_complexity == "low":
            base_config["model"] = "gpt-3.5-turbo"
            base_config["max_tokens"] = 500
            
        return base_config 