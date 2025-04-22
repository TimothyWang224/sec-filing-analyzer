"""
Agent Workflow Page

This page provides a user interface for interacting with the agent workflow.
"""

import streamlit as st
import pandas as pd
import asyncio
import sys
from pathlib import Path
import uuid

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Import agent components
try:
    from sec_filing_analyzer.config import ConfigProvider
    from sec_filing_analyzer.llm.llm_config import LLMConfigFactory, get_agent_types
    from src.agents import (
        FinancialAnalystAgent,
        RiskAnalystAgent,
        QASpecialistAgent,
        FinancialDiligenceCoordinator
    )
    from src.environments import FinancialEnvironment
    from src.capabilities import TimeAwarenessCapability, LoggingCapability
    imports_successful = True
except ImportError as e:
    st.error(f"Error importing SEC Filing Analyzer components: {e}")
    st.warning("Some functionality may be limited. Please make sure the SEC Filing Analyzer package is installed correctly.")
    imports_successful = False

    # Define fallback functions and classes
    def get_agent_types():
        return ["coordinator", "financial_analyst", "risk_analyst", "qa_specialist"]

    class FallbackConfigFactory:
        @staticmethod
        def get_available_models():
            return {"gpt-4o": "OpenAI GPT-4o", "gpt-4o-mini": "OpenAI GPT-4o Mini", "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo"}

        @staticmethod
        def get_recommended_config(**kwargs):
            # Ignore parameters, just return default config
            _ = kwargs  # Suppress unused variable warning
            return {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 4000}

# Set page config
st.set_page_config(
    page_title="Agent Workflow - SEC Filing Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize configuration
if imports_successful:
    try:
        ConfigProvider.initialize()
    except Exception as config_error:
        st.error(f"Error initializing configuration: {config_error}")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False

# Title and description
st.title("Agent Workflow")
st.markdown("""
Interact with intelligent agents to analyze SEC filings data.
Choose an agent type and start a conversation to analyze financial data.
""")

# Sidebar for agent configuration
st.sidebar.header("Agent Configuration")

# Agent type selection
st.sidebar.subheader("Agent Type")
if imports_successful:
    try:
        agent_types = get_agent_types()
    except Exception as e:
        st.error(f"Error getting agent types: {e}")
        agent_types = ["coordinator", "financial_analyst", "risk_analyst", "qa_specialist"]
else:
    agent_types = get_agent_types()

agent_type = st.sidebar.selectbox(
    "Select Agent Type",
    agent_types,
    index=agent_types.index("coordinator") if "coordinator" in agent_types else 0
)

# LLM model selection
st.sidebar.subheader("LLM Configuration")
if imports_successful:
    try:
        available_models = LLMConfigFactory.get_available_models()
    except Exception as e:
        st.error(f"Error getting available models: {e}")
        available_models = {"gpt-4o": "OpenAI GPT-4o", "gpt-4o-mini": "OpenAI GPT-4o Mini", "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo"}
else:
    available_models = FallbackConfigFactory.get_available_models()

model_options = list(available_models.keys())
model_descriptions = [f"{model} - {desc}" for model, desc in available_models.items()]

selected_model_index = st.sidebar.selectbox(
    "Select LLM Model",
    range(len(model_options)),
    format_func=lambda i: model_descriptions[i]
)
selected_model = model_options[selected_model_index]

# Temperature setting
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1
)

# Max tokens setting
max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=500,
    max_value=8000,
    value=4000,
    step=500
)

# Task complexity
task_complexity = st.sidebar.radio(
    "Task Complexity",
    ["low", "medium", "high"],
    index=1
)

# Advanced agent settings
with st.sidebar.expander("Advanced Settings"):
    max_iterations = st.number_input("Max Iterations", min_value=1, max_value=10, value=3)
    max_planning_iterations = st.number_input("Max Planning Iterations", min_value=1, max_value=5, value=2)
    max_execution_iterations = st.number_input("Max Execution Iterations", min_value=1, max_value=5, value=3)
    max_refinement_iterations = st.number_input("Max Refinement Iterations", min_value=1, max_value=5, value=1)
    max_tool_retries = st.number_input("Max Tool Retries", min_value=1, max_value=5, value=2)
    max_duration_seconds = st.number_input("Max Duration (seconds)", min_value=30, max_value=600, value=180)

# Initialize agent
if st.sidebar.button("Initialize Agent"):
    with st.spinner("Initializing agent..."):
        if not imports_successful:
            st.error("Cannot initialize agent: Required components are not available.")
            st.info("Please make sure the SEC Filing Analyzer package is installed correctly using 'poetry install'.")
            st.session_state.agent_initialized = False
        else:
            try:
                # Get recommended configuration
                try:
                    config = LLMConfigFactory.get_recommended_config(
                        agent_type=agent_type,
                        task_complexity=task_complexity
                    )
                except Exception as config_error:
                    st.warning(f"Error getting recommended config: {config_error}")
                    # Use default config
                    config = {
                        "model": selected_model,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }

                # Override with user settings
                config.update({
                    "model": selected_model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "max_iterations": max_iterations,
                    "max_planning_iterations": max_planning_iterations,
                    "max_execution_iterations": max_execution_iterations,
                    "max_refinement_iterations": max_refinement_iterations,
                    "max_tool_retries": max_tool_retries,
                    "max_duration_seconds": max_duration_seconds
                })

                # Initialize environment
                environment = None
                try:
                    environment = FinancialEnvironment()
                except Exception as env_error:
                    st.error(f"Error initializing environment: {env_error}")
                    st.session_state.agent_initialized = False

                # Skip the rest of the initialization if environment failed
                if environment is None:
                    # Cannot use break or continue here, so we'll just set a flag
                    st.error("Cannot proceed without a valid environment.")
                    st.session_state.agent_initialized = False
                else:
                    # Initialize capabilities
                    try:
                        capabilities = [
                            TimeAwarenessCapability(),
                            LoggingCapability()
                        ]
                    except Exception as cap_error:
                        st.error(f"Error initializing capabilities: {cap_error}")
                        capabilities = []

                    # Initialize agent based on type
                    try:
                        if agent_type == "coordinator":
                            st.session_state.agent = FinancialDiligenceCoordinator(
                                environment=environment,
                                capabilities=capabilities,
                                **config
                            )
                        elif agent_type == "financial_analyst":
                            st.session_state.agent = FinancialAnalystAgent(
                                environment=environment,
                                capabilities=capabilities,
                                **config
                            )
                        elif agent_type == "risk_analyst":
                            st.session_state.agent = RiskAnalystAgent(
                                environment=environment,
                                capabilities=capabilities,
                                **config
                            )
                        elif agent_type == "qa_specialist":
                            st.session_state.agent = QASpecialistAgent(
                                environment=environment,
                                capabilities=capabilities,
                                **config
                            )

                        st.session_state.agent_initialized = True
                        st.success(f"{agent_type.title()} agent initialized successfully!")

                        # Clear chat history when initializing a new agent
                        st.session_state.messages = []
                    except Exception as agent_error:
                        st.error(f"Error initializing agent: {agent_error}")
                        st.session_state.agent_initialized = False

            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.session_state.agent_initialized = False

# Main content - Chat interface
st.header("Agent Conversation")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new message
if st.session_state.agent_initialized:
    user_input = st.chat_input("Type your message here...")

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
                # Run the agent
                response = asyncio.run(st.session_state.agent.run(user_input))

                # Extract the response content
                if isinstance(response, dict) and "response" in response:
                    agent_response = response["response"]
                else:
                    agent_response = str(response)

                # Update the message placeholder
                message_placeholder.markdown(agent_response)

                # Add assistant message to chat
                st.session_state.messages.append({"role": "assistant", "content": agent_response})

            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("Please initialize an agent to start the conversation.")

# Conversation controls
st.header("Conversation Controls")

col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("Export Conversation"):
        # Create a DataFrame from the conversation
        conversation_df = pd.DataFrame(st.session_state.messages)

        # Convert to CSV
        csv = conversation_df.to_csv(index=False).encode('utf-8')

        # Create a download button
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"conversation_{uuid.uuid4().hex[:8]}.csv",
            mime="text/csv"
        )
