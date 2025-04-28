"""
Test script to verify the ConfigProvider functionality.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.sec_filing_analyzer.config import ConfigProvider
    from src.sec_filing_analyzer.llm.llm_config import LLMConfigFactory

    # Initialize the ConfigProvider
    ConfigProvider.initialize()

    # Test getting agent types
    print("Available agent types:")
    agent_types = ConfigProvider.get_all_agent_types()
    print(agent_types)
    print()

    # Test getting agent configs
    for agent_type in agent_types:
        print(f"Configuration for {agent_type}:")
        config = ConfigProvider.get_agent_config(agent_type)

        # Print key parameters
        print(f"  Model: {config.get('model', 'Not specified')}")
        print(f"  Temperature: {config.get('temperature', 'Not specified')}")
        print(f"  Max Tokens: {config.get('max_tokens', 'Not specified')}")
        print(f"  Max Iterations: {config.get('max_iterations', 'Not specified')}")
        print(
            f"  Max Planning Iterations: {config.get('max_planning_iterations', 'Not specified')}"
        )
        print(
            f"  Max Execution Iterations: {config.get('max_execution_iterations', 'Not specified')}"
        )
        print(
            f"  Max Refinement Iterations: {config.get('max_refinement_iterations', 'Not specified')}"
        )
        print()

    # Test getting agent config via LLMConfigFactory
    print("Testing LLMConfigFactory:")
    for agent_type in agent_types:
        print(f"  Configuration for {agent_type} via LLMConfigFactory:")
        config = LLMConfigFactory.create_config_from_provider(agent_type)
        print(f"    Model: {config.get('model', 'Not specified')}")
        print(f"    Temperature: {config.get('temperature', 'Not specified')}")
        print()

    # Test getting agent config with different task complexities
    print("Testing task complexity configurations:")
    for complexity in ["low", "medium", "high"]:
        print(f"  Configuration for coordinator with {complexity} complexity:")
        config = LLMConfigFactory.get_recommended_config(
            "coordinator", task_complexity=complexity
        )
        print(f"    Model: {config.get('model', 'Not specified')}")
        print(f"    Max Tokens: {config.get('max_tokens', 'Not specified')}")
        print(
            f"    Max Planning Iterations: {config.get('max_planning_iterations', 'Not specified')}"
        )
        print()

    print("All tests completed successfully!")

except Exception as e:
    print(f"Error testing ConfigProvider: {str(e)}")
    import traceback

    traceback.print_exc()
