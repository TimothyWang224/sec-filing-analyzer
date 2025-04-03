# SEC Filing Analyzer

A multi-agent LLM system for financial analysis and decision support, specializing in SEC filing analysis, financial Q&A, and risk monitoring.

## Features

- **Automated Financial Diligence**: Agents review SEC filings, extract key financial ratios, company risks, and track changes between quarterly reports.
- **Real-time Financial Q&A**: Interactive system for detailed financial questions about revenue growth, leverage changes, and management disclosures.
- **Enhanced Risk Monitoring**: LLM agents monitor, summarize, and alert on material financial and operational risks from structured and unstructured financial documents.

## Project Structure

```
sec-filing-analyzer/
├── src/                    # Source code
│   ├── agents/            # LLM agents for different analysis tasks
│   ├── capabilities/      # Agent capabilities and behaviors
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

# Initialize the coordinator
coordinator = FinancialDiligenceCoordinator()

# Analyze a company
results = await coordinator.analyze_company("AAPL")
```

### Financial Q&A

```python
from src.agents.qa_specialist import QASpecialistAgent

# Initialize the Q&A agent
qa_agent = QASpecialistAgent()

# Ask a question
response = await qa_agent.answer_question(
    "What's driving Apple's revenue growth in the latest quarter?"
)
```

### Risk Monitoring

```python
from src.agents.risk_analyst import RiskAnalystAgent

# Initialize the risk analyst
risk_agent = RiskAnalystAgent()

# Monitor risks
risks = await risk_agent.monitor_risks("AAPL")
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Adding New Agents

1. Create a new agent class in `src/agents/`
2. Define its goals and capabilities
3. Register its tools
4. Add tests in `tests/`

### Adding New Tools

1. Create a new tool in `src/tools/`
2. Register it with the appropriate agent
3. Add tests in `tests/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
