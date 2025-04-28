"""
SEC Filing Analyzer Chat Demo App

This Streamlit app provides a simplified chat interface for interacting with the
SEC Filing Analyzer in demo mode.
"""

import streamlit as st

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="SEC Filing Analyzer - Demo Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

import asyncio
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for demo mode
DEMO_MODE = os.getenv("SFA_DEMO_MODE", "0") == "1"
if not DEMO_MODE:
    os.environ["SFA_DEMO_MODE"] = "1"
    DEMO_MODE = True

# Import agent components
imports_successful = True
try:
    from sec_filing_analyzer.config import ConfigProvider
    from sec_filing_analyzer.llm.llm_config import LLMConfigFactory

    # Import the SimpleChatAgent
    from examples.simple_chat_agent import SimpleChatAgent
    
    # Import capabilities
    from src.capabilities import (
        LoggingCapability,
        TimeAwarenessCapability,
    )
    
    # Import environment
    from src.environments import FinancialEnvironment
    
    # Import tools
    from src.tools.sec_financial_data import SECFinancialDataTool
    from src.tools.sec_semantic_search import SECSemanticSearchTool
    from src.tools.sec_data import SECDataTool
    
except ImportError as e:
    st.error(f"Error importing SEC Filing Analyzer components: {e}")
    st.warning(
        "Some functionality may be limited. Please make sure the SEC Filing Analyzer package is installed correctly."
    )
    imports_successful = False

# Initialize configuration
if imports_successful:
    try:
        ConfigProvider.initialize()
    except Exception as config_error:
        st.error(f"Error initializing configuration: {config_error}")

# Title and description
st.title("SEC Filing Analyzer Demo Chat")
st.markdown("""
This is a lightweight demo version of the SEC Filing Analyzer Chat.
Ask questions about companies, financial metrics, and SEC filings.

**Demo Features:**
- Search for information in SEC filings
- Query financial metrics and facts
- Get information about companies and filings

**Sample Questions:**
- "What was NVDA's revenue in 2023?"
- "Find information about Apple's AI strategy"
- "What are the risk factors for Microsoft?"
""")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None
    st.session_state.agent_initialized = False

if "config" not in st.session_state:
    st.session_state.config = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 4000,
    }

# Sidebar for configuration
st.sidebar.header("Demo Configuration")

# Model selection
if imports_successful:
    try:
        available_models = LLMConfigFactory.get_available_models()
    except Exception as e:
        st.sidebar.error(f"Error getting available models: {e}")
        available_models = {
            "gpt-4o": "OpenAI GPT-4o",
            "gpt-4o-mini": "OpenAI GPT-4o Mini",
            "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo",
        }
else:
    available_models = {
        "gpt-4o": "OpenAI GPT-4o",
        "gpt-4o-mini": "OpenAI GPT-4o Mini",
        "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo",
    }

model_options = list(available_models.keys())
model_descriptions = [f"{model} - {desc}" for model, desc in available_models.items()]

selected_model_index = st.sidebar.selectbox(
    "LLM Model",
    range(len(model_options)),
    format_func=lambda i: model_descriptions[i],
    index=model_options.index(st.session_state.config["model"])
    if st.session_state.config["model"] in model_options
    else 0,
)
selected_model = model_options[selected_model_index]

# Temperature slider
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.config["temperature"],
    step=0.1,
    help="Higher values make output more random, lower values more deterministic",
)

# Max tokens slider
max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=1000,
    max_value=8000,
    value=st.session_state.config["max_tokens"],
    step=1000,
    help="Maximum number of tokens in the response",
)

# Update configuration
st.session_state.config["model"] = selected_model
st.session_state.config["temperature"] = temperature
st.session_state.config["max_tokens"] = max_tokens

# Demo data path configuration
demo_vector_store_path = st.sidebar.text_input(
    "Demo Vector Store Path",
    value="data/demo_assets/vector_demo",
    help="Path to the demo vector store (leave as is unless you know what you're doing)",
)

# Function to initialize the simple chat agent
def initialize_simple_chat_agent():
    """Initialize the simple chat agent with capabilities."""
    if not imports_successful:
        st.error("Cannot initialize agent: Required components are not available.")
        return None

    try:
        # Create environment
        environment = FinancialEnvironment()

        # Create capabilities
        capabilities = [
            TimeAwarenessCapability(),
            LoggingCapability(
                include_prompts=True,
                include_responses=True,
                max_content_length=10000,
            ),
        ]

        # Create tools
        tools = [
            SECSemanticSearchTool(vector_store_path=demo_vector_store_path),
            SECFinancialDataTool(),
            SECDataTool(),
        ]

        # Create simple chat agent
        agent = SimpleChatAgent(
            environment=environment,
            capabilities=capabilities,
            tools=tools,
            llm_model=st.session_state.config["model"],
            llm_temperature=st.session_state.config["temperature"],
            llm_max_tokens=st.session_state.config["max_tokens"],
            max_iterations=5,
            max_tool_retries=2,
            tools_per_iteration=1,
            vector_store_path=demo_vector_store_path,
        )

        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None


# Initialize agent button
if st.sidebar.button("Initialize Agent"):
    with st.spinner("Initializing agent..."):
        st.session_state.agent = initialize_simple_chat_agent()
        if st.session_state.agent:
            st.session_state.agent_initialized = True
            st.sidebar.success("Agent initialized successfully!")
        else:
            st.sidebar.error("Failed to initialize agent.")

# Reset chat button
if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []
    st.rerun()

# Update agent button
if st.session_state.agent_initialized and st.sidebar.button(
    "Update Agent Configuration"
):
    with st.spinner("Updating agent configuration..."):
        # Create a new agent with updated configuration
        st.session_state.agent = initialize_simple_chat_agent()
        if st.session_state.agent:
            st.sidebar.success("Agent configuration updated!")
        else:
            st.sidebar.error("Failed to update agent configuration.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new message
if st.session_state.agent_initialized:
    user_input = st.chat_input("Ask a question about SEC filings...")

    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get agent response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            try:
                # Run the agent with chat mode enabled
                start_time = time.time()
                response = asyncio.run(
                    st.session_state.agent.run(user_input, chat_mode=True)
                )
                elapsed_time = time.time() - start_time

                # Extract the response content
                if isinstance(response, dict):
                    if "response" in response:
                        # Use the formatted response from chat mode
                        agent_response = response["response"]
                    else:
                        agent_response = str(response)
                else:
                    agent_response = str(response)

                # Add processing time information
                agent_response += (
                    f"\n\n*Response generated in {elapsed_time:.2f} seconds*"
                )

                # Update the message placeholder
                message_placeholder.markdown(agent_response)

                # Add assistant message to chat
                st.session_state.messages.append(
                    {"role": "assistant", "content": agent_response}
                )

            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
else:
    st.info(
        "Please initialize the agent using the button in the sidebar to start chatting."
    )

# Demo notice
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Demo Mode
This is running in demo mode with limited functionality.
For the full experience, use the complete SEC Filing Analyzer.
""")
