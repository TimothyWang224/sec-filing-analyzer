import logging

from src.agents.coordinator import FinancialDiligenceCoordinator
from src.sec_filing_analyzer.config import AgentConfig, ConfigProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_agent_initialization():
    """Test initializing the FinancialDiligenceCoordinator agent."""
    try:
        logger.info("Initializing FinancialDiligenceCoordinator agent...")

        # Get agent configuration
        agent_config = ConfigProvider.get_config(AgentConfig)

        # Initialize the agent
        agent = FinancialDiligenceCoordinator(
            name="FinancialDiligenceCoordinator", config=agent_config
        )

        logger.info("Agent initialized successfully!")
        return agent
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    agent = test_agent_initialization()
    if agent:
        print("Agent initialized successfully!")
    else:
        print("Failed to initialize agent. Check the logs for details.")
