# SEC Filing Analyzer

A multi-agent LLM system for financial analysis and decision support, specializing in SEC filing analysis, financial Q&A, and risk monitoring.

## Features

- **Automated Financial Diligence**: Agents review SEC filings, extract key financial ratios, company risks, and track changes between quarterly reports.
- **Real-time Financial Q&A**: Interactive system for detailed financial questions about revenue growth, leverage changes, and management disclosures.
- **Enhanced Risk Monitoring**: LLM agents monitor, summarize, and alert on material financial and operational risks from structured and unstructured financial documents.
- **Flexible LLM Configuration**: Configurable LLM settings per agent type with support for different models and parameters.
- **Modular Architecture**: Registry-based system for easy addition of new agents and capabilities.

## Project Structure

```
sec-filing-analyzer/
├── src/                    # Source code
│   ├── agents/            # LLM agents for different analysis tasks
│   │   ├── base.py       # Base agent implementation
│   │   ├── registry.py   # Agent registration system
│   │   └── ...          # Specialized agents
│   ├── capabilities/      # Agent capabilities and behaviors
│   │   ├── base.py      # Base capability implementation
│   │   ├── registry.py  # Capability registration system
│   │   └── ...         # Specialized capabilities
│   ├── sec_filing_analyzer/
│   │   └── llm/         # LLM integration
│   │       ├── base.py  # Base LLM interface
│   │       ├── openai.py # OpenAI implementation
│   │       └── llm_config.py # LLM configuration management
│   ├── tools/            # Tools available to agents
│   ├── memory/           # Memory management for agents
│   ├── environments/     # Agent environments
│   └── api/              # API endpoints
├── tests/                 # Test suite
├── docs/                  # Documentation
└── examples/              # Example usage scripts
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sec-filing-analyzer.git
cd sec-filing-analyzer
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Basic Example

```python
from src.agents.coordinator import FinancialDiligenceCoordinator
from src.sec_filing_analyzer.llm import LLMConfigFactory

# Initialize the coordinator with custom LLM config
llm_config = LLMConfigFactory.create_config(
    "coordinator",
    model="gpt-4-turbo-preview",
    temperature=0.7
)
coordinator = FinancialDiligenceCoordinator(llm_config=llm_config)

# Analyze a company
results = await coordinator.analyze_company("AAPL")
```

### Financial Q&A

```python
from src.agents.qa_specialist import QASpecialistAgent
from src.sec_filing_analyzer.llm import LLMConfigFactory

# Initialize the Q&A agent with recommended config
llm_config = LLMConfigFactory.get_recommended_config(
    "qa_specialist",
    task_complexity="high"
)
qa_agent = QASpecialistAgent(llm_config=llm_config)

# Ask a question
response = await qa_agent.answer_question(
    "What's driving Apple's revenue growth in the latest quarter?"
)
```

### Risk Monitoring

```python
from src.agents.risk_analyst import RiskAnalystAgent
from src.capabilities import CapabilityRegistry

# Initialize the risk analyst with specific capabilities
risk_agent = RiskAnalystAgent(
    capabilities=CapabilityRegistry.create_capabilities(["sec_analysis"])
)

# Monitor risks
risks = await risk_agent.monitor_risks("AAPL")
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Adding New Agents

1. Create a new agent class in `src/agents/`:
```python
from src.agents.base import Agent, Goal
from src.sec_filing_analyzer.llm import LLMConfigFactory

class NewAgent(Agent):
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        goals = [Goal(name="new_goal", description="...")]
        llm_config = llm_config or LLMConfigFactory.create_config("new_agent")
        super().__init__(goals=goals, llm_config=llm_config)
```

2. Register the agent in `src/agents/__init__.py`:
```python
from .registry import AgentRegistry
AgentRegistry.register("new_agent", NewAgent)
```

3. Add LLM configuration in `src/sec_filing_analyzer/llm/llm_config.py`:
```python
NEW_AGENT_CONFIG = {
    **BASE_CONFIG,
    "model": "gpt-3.5-turbo",
    "temperature": 0.5,
    "system_prompt": "You are a new agent..."
}

AGENT_CONFIGS["new_agent"] = NEW_AGENT_CONFIG
```

4. Add tests in `tests/`

### Adding New Capabilities

1. Create a new capability in `src/capabilities/`:
```python
from src.capabilities.base import Capability

class NewCapability(Capability):
    def __init__(self):
        super().__init__(name="new_capability")
```

2. Register the capability in `src/capabilities/__init__.py`:
```python
from .registry import CapabilityRegistry
CapabilityRegistry.register("new_capability", NewCapability)
```

3. Add tests in `tests/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
