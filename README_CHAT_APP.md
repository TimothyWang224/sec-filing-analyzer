# SEC Filing Analyzer Chat App

This is a dedicated chat application for interacting with the SEC Filing Analyzer's coordinator agent. The app provides a clean, focused interface for asking questions about SEC filings and receiving comprehensive analyses.

## Features

- **Conversational Interface**: Chat directly with the Financial Diligence Coordinator agent
- **Comprehensive Analysis**: The coordinator delegates to specialized agents as needed
- **Configurable**: Adjust LLM model, temperature, and other parameters
- **Persistent Chat History**: Chat history is maintained during the session

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Poetry (for dependency management)
- SEC Filing Analyzer package installed

### Installation

1. Make sure you have Poetry installed. If not, follow the instructions at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation).

2. Set up your OpenAI API key in the `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Chat App

1. Run the chat app using the provided batch file (Windows):
   ```
   run_chat_app.bat
   ```

   Or using the shell script (Linux/Mac):
   ```
   chmod +x run_chat_app.sh
   ./run_chat_app.sh
   ```

   The scripts will automatically install the required dependencies using Poetry and launch the app.

2. The app will open in your default web browser at `http://localhost:8501`

## Usage

1. **Initialize the Agent**: Click the "Initialize Agent" button in the sidebar to start the agent.

2. **Ask Questions**: Type your questions or requests in the chat input field at the bottom of the page.

3. **Configure the Agent**: Use the sidebar to adjust:
   - LLM Model: Select from available models
   - Temperature: Adjust the randomness of responses
   - Max Tokens: Set the maximum response length

4. **Reset Chat**: Click "Reset Chat" to clear the conversation history.

5. **Update Configuration**: Click "Update Agent Configuration" to apply changes to the agent.

## Example Questions

- "What was Apple's revenue growth in 2023?"
- "Analyze Microsoft's financial performance over the last 3 years."
- "What are the main risks mentioned in NVIDIA's latest 10-K?"
- "Compare Google and Apple's profit margins."
- "Summarize the key financial metrics for NVIDIA."

## Architecture

The chat app is built on:

1. **Streamlit**: For the web interface
2. **Financial Diligence Coordinator**: The main agent that orchestrates analysis
3. **Specialized Agents**: Financial Analyst, Risk Analyst, and QA Specialist
4. **Planning Capability**: For breaking down complex requests into manageable steps

## Standalone App

This chat app is implemented as a standalone Streamlit application, separate from the ETL and data exploration functionality. It's located in the `chat_app` directory and has its own entry point. This separation provides:

1. A cleaner, more focused user experience
2. Independence from the ETL pipeline components
3. Simplified navigation without additional pages

## Troubleshooting

- **Agent Initialization Fails**: Check your OpenAI API key and internet connection.
- **Slow Responses**: Complex analyses may take time, especially with larger models.
- **Error Messages**: If you receive an error, try rephrasing your question or check the logs.

## Separate from ETL App

This chat app is intentionally separate from the ETL and data exploration functionality to:
1. Provide a cleaner, more focused user experience
2. Allow independent development and maintenance
3. Optimize the interface for conversational interaction
