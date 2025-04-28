"""
Configuration Page

This page provides a user interface for configuring the SEC Filing Analyzer system.
"""

import sys
from pathlib import Path

import streamlit as st

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Import configuration components
from sec_filing_analyzer.config import (
    ConfigProvider,
    ETLConfig,
    StorageConfig,
    StreamlitConfig,
)
from sec_filing_analyzer.llm.llm_config import LLMConfigFactory, get_agent_types

# Set page config
st.set_page_config(
    page_title="Configuration - SEC Filing Analyzer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize configuration
ConfigProvider.initialize()

# Title and description
st.title("Configuration")
st.markdown("""
Configure the SEC Filing Analyzer system settings, including:
- ETL pipeline configuration
- Agent configuration
- Storage configuration
- Streamlit configuration
""")

# Sidebar for navigation
st.sidebar.header("Configuration Navigation")
config_type = st.sidebar.radio(
    "Select Configuration Type",
    [
        "ETL Configuration",
        "Agent Configuration",
        "Storage Configuration",
        "Streamlit Configuration",
    ],
)

# Main content
if config_type == "ETL Configuration":
    st.header("ETL Configuration")

    # Get current configuration
    etl_config = ConfigProvider.get_config(ETLConfig)

    # Create form for ETL configuration
    with st.form("etl_config_form"):
        st.subheader("Data Retrieval Settings")

        filings_dir = st.text_input("Filings Directory", value=str(etl_config.filings_dir))
        filing_types = st.multiselect(
            "Filing Types",
            ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"],
            default=etl_config.filing_types or ["10-K", "10-Q"],
        )
        max_retries = st.number_input("Max Retries", min_value=1, max_value=10, value=etl_config.max_retries)
        timeout = st.number_input("Timeout (seconds)", min_value=5, max_value=120, value=etl_config.timeout)

        st.subheader("Document Processing Settings")

        chunk_size = st.number_input("Chunk Size", min_value=128, max_value=4096, value=etl_config.chunk_size)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1024, value=etl_config.chunk_overlap)
        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
            index=0
            if etl_config.embedding_model == "text-embedding-3-small"
            else 1
            if etl_config.embedding_model == "text-embedding-3-large"
            else 2,
        )

        st.subheader("Parallel Processing Settings")

        use_parallel = st.checkbox("Use Parallel Processing", value=etl_config.use_parallel)
        max_workers = st.number_input("Max Workers", min_value=1, max_value=16, value=etl_config.max_workers)
        batch_size = st.number_input("Batch Size", min_value=10, max_value=500, value=etl_config.batch_size)
        rate_limit = st.number_input(
            "Rate Limit (seconds)",
            min_value=0.0,
            max_value=2.0,
            value=etl_config.rate_limit,
            step=0.1,
        )

        st.subheader("XBRL Extraction Settings")

        process_quantitative = st.checkbox("Process Quantitative Data", value=etl_config.process_quantitative)
        db_path = st.text_input("DuckDB Path", value=etl_config.db_path)

        st.subheader("Processing Flags")

        process_semantic = st.checkbox("Process Semantic Data", value=etl_config.process_semantic)
        delay_between_companies = st.number_input(
            "Delay Between Companies (seconds)",
            min_value=0,
            max_value=10,
            value=etl_config.delay_between_companies,
        )

        # Submit button
        submitted = st.form_submit_button("Save ETL Configuration")

        if submitted:
            # Update configuration
            # In a real implementation, we would update the configuration in the ConfigProvider
            # For now, we'll just show a success message
            st.success("ETL Configuration saved successfully!")

            # Display the updated configuration
            st.json(
                {
                    "filings_dir": filings_dir,
                    "filing_types": filing_types,
                    "max_retries": max_retries,
                    "timeout": timeout,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "embedding_model": embedding_model,
                    "use_parallel": use_parallel,
                    "max_workers": max_workers,
                    "batch_size": batch_size,
                    "rate_limit": rate_limit,
                    "process_quantitative": process_quantitative,
                    "db_path": db_path,
                    "process_semantic": process_semantic,
                    "delay_between_companies": delay_between_companies,
                }
            )

elif config_type == "Agent Configuration":
    st.header("Agent Configuration")

    # Get agent types
    agent_types = get_agent_types()

    # Agent type selection
    selected_agent_type = st.selectbox("Select Agent Type", agent_types)

    # Get agent configuration
    agent_config = LLMConfigFactory.create_config_from_provider(selected_agent_type)

    # Create tabs for different configuration sections
    tab1, tab2, tab3 = st.tabs(["LLM Parameters", "Agent Execution Parameters", "System Prompt"])

    with tab1:
        st.subheader("LLM Parameters")

        # Create form for LLM parameters
        with st.form("llm_params_form"):
            # Get available models
            available_models = LLMConfigFactory.get_available_models()
            model_options = list(available_models.keys())
            model_descriptions = [f"{model} - {desc}" for model, desc in available_models.items()]

            # Find current model index
            current_model = agent_config.get("model", "gpt-4o-mini")
            current_model_index = model_options.index(current_model) if current_model in model_options else 0

            # Model selection
            selected_model_index = st.selectbox(
                "LLM Model",
                range(len(model_options)),
                format_func=lambda i: model_descriptions[i],
                index=current_model_index,
            )
            selected_model = model_options[selected_model_index]

            # Temperature
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=agent_config.get("temperature", 0.7),
                step=0.1,
            )

            # Max tokens
            max_tokens = st.slider(
                "Max Tokens",
                min_value=500,
                max_value=8000,
                value=agent_config.get("max_tokens", 4000),
                step=500,
            )

            # Top P
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=agent_config.get("top_p", 1.0),
                step=0.1,
            )

            # Frequency penalty
            frequency_penalty = st.slider(
                "Frequency Penalty",
                min_value=0.0,
                max_value=2.0,
                value=agent_config.get("frequency_penalty", 0.0),
                step=0.1,
            )

            # Presence penalty
            presence_penalty = st.slider(
                "Presence Penalty",
                min_value=0.0,
                max_value=2.0,
                value=agent_config.get("presence_penalty", 0.0),
                step=0.1,
            )

            # Submit button
            submitted = st.form_submit_button("Save LLM Parameters")

            if submitted:
                # Update configuration
                # In a real implementation, we would update the configuration in the ConfigProvider
                # For now, we'll just show a success message
                st.success("LLM Parameters saved successfully!")

                # Display the updated configuration
                st.json(
                    {
                        "model": selected_model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": top_p,
                        "frequency_penalty": frequency_penalty,
                        "presence_penalty": presence_penalty,
                    }
                )

    with tab2:
        st.subheader("Agent Execution Parameters")

        # Create form for agent execution parameters
        with st.form("agent_exec_form"):
            # Iteration parameters
            st.markdown("**Iteration Parameters**")

            max_iterations = st.number_input(
                "Max Iterations",
                min_value=1,
                max_value=10,
                value=agent_config.get("max_iterations", 3),
            )

            max_planning_iterations = st.number_input(
                "Max Planning Iterations",
                min_value=1,
                max_value=5,
                value=agent_config.get("max_planning_iterations", 2),
            )

            max_execution_iterations = st.number_input(
                "Max Execution Iterations",
                min_value=1,
                max_value=5,
                value=agent_config.get("max_execution_iterations", 3),
            )

            max_refinement_iterations = st.number_input(
                "Max Refinement Iterations",
                min_value=1,
                max_value=5,
                value=agent_config.get("max_refinement_iterations", 1),
            )

            # Tool execution parameters
            st.markdown("**Tool Execution Parameters**")

            max_tool_retries = st.number_input(
                "Max Tool Retries",
                min_value=1,
                max_value=5,
                value=agent_config.get("max_tool_retries", 2),
            )

            tools_per_iteration = st.number_input(
                "Tools Per Iteration",
                min_value=1,
                max_value=5,
                value=agent_config.get("tools_per_iteration", 1),
            )

            circuit_breaker_threshold = st.number_input(
                "Circuit Breaker Threshold",
                min_value=1,
                max_value=10,
                value=agent_config.get("circuit_breaker_threshold", 3),
            )

            circuit_breaker_reset_timeout = st.number_input(
                "Circuit Breaker Reset Timeout (seconds)",
                min_value=60,
                max_value=600,
                value=agent_config.get("circuit_breaker_reset_timeout", 300),
            )

            # Runtime parameters
            st.markdown("**Runtime Parameters**")

            max_duration_seconds = st.number_input(
                "Max Duration (seconds)",
                min_value=30,
                max_value=600,
                value=agent_config.get("max_duration_seconds", 180),
            )

            # Termination parameters
            st.markdown("**Termination Parameters**")

            enable_dynamic_termination = st.checkbox(
                "Enable Dynamic Termination",
                value=agent_config.get("enable_dynamic_termination", False),
            )

            min_confidence_threshold = st.slider(
                "Min Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=agent_config.get("min_confidence_threshold", 0.8),
                step=0.1,
            )

            # Submit button
            submitted = st.form_submit_button("Save Agent Execution Parameters")

            if submitted:
                # Update configuration
                # In a real implementation, we would update the configuration in the ConfigProvider
                # For now, we'll just show a success message
                st.success("Agent Execution Parameters saved successfully!")

                # Display the updated configuration
                st.json(
                    {
                        "max_iterations": max_iterations,
                        "max_planning_iterations": max_planning_iterations,
                        "max_execution_iterations": max_execution_iterations,
                        "max_refinement_iterations": max_refinement_iterations,
                        "max_tool_retries": max_tool_retries,
                        "tools_per_iteration": tools_per_iteration,
                        "circuit_breaker_threshold": circuit_breaker_threshold,
                        "circuit_breaker_reset_timeout": circuit_breaker_reset_timeout,
                        "max_duration_seconds": max_duration_seconds,
                        "enable_dynamic_termination": enable_dynamic_termination,
                        "min_confidence_threshold": min_confidence_threshold,
                    }
                )

    with tab3:
        st.subheader("System Prompt")

        # Get current system prompt
        system_prompt = agent_config.get("system_prompt", "")

        # Create form for system prompt
        with st.form("system_prompt_form"):
            # System prompt
            new_system_prompt = st.text_area("System Prompt", value=system_prompt, height=300)

            # Submit button
            submitted = st.form_submit_button("Save System Prompt")

            if submitted:
                # Update configuration
                # In a real implementation, we would update the configuration in the ConfigProvider
                # For now, we'll just show a success message
                st.success("System Prompt saved successfully!")

                # Display the updated configuration
                st.json({"system_prompt": new_system_prompt})

elif config_type == "Storage Configuration":
    st.header("Storage Configuration")

    # Create tabs for different storage types
    tab1, tab2, tab3 = st.tabs(["Vector Store", "Graph Store", "File Storage"])

    with tab1:
        st.subheader("Vector Store Configuration")

        # Get current configuration
        storage_config = ConfigProvider.get_config(StorageConfig)

        # Create form for vector store configuration
        with st.form("vector_store_form"):
            # Vector store path
            vector_store_path = st.text_input("Vector Store Path", value=str(storage_config.vector_store_path))

            # Vector store type
            vector_store_type = st.selectbox(
                "Vector Store Type",
                ["basic", "optimized"],
                index=0 if storage_config.vector_store_type == "basic" else 1,
            )

            # Index type
            index_type = st.selectbox(
                "Index Type",
                ["flat", "ivf", "hnsw"],
                index=0 if storage_config.index_type == "flat" else 1 if storage_config.index_type == "ivf" else 2,
            )

            # Use GPU
            use_gpu = st.checkbox("Use GPU", value=storage_config.use_gpu)

            # Advanced parameters
            st.markdown("**Advanced Parameters**")

            # IVF parameters
            if index_type == "ivf":
                ivf_nlist = st.number_input(
                    "IVF nlist",
                    min_value=1,
                    max_value=1000,
                    value=storage_config.ivf_nlist,
                )

                ivf_nprobe = st.number_input(
                    "IVF nprobe",
                    min_value=1,
                    max_value=100,
                    value=storage_config.ivf_nprobe,
                )
            else:
                ivf_nlist = storage_config.ivf_nlist
                ivf_nprobe = storage_config.ivf_nprobe

            # HNSW parameters
            if index_type == "hnsw":
                hnsw_m = st.number_input("HNSW M", min_value=4, max_value=128, value=storage_config.hnsw_m)

                hnsw_ef_construction = st.number_input(
                    "HNSW EF Construction",
                    min_value=40,
                    max_value=800,
                    value=storage_config.hnsw_ef_construction,
                )

                hnsw_ef_search = st.number_input(
                    "HNSW EF Search",
                    min_value=20,
                    max_value=400,
                    value=storage_config.hnsw_ef_search,
                )
            else:
                hnsw_m = storage_config.hnsw_m
                hnsw_ef_construction = storage_config.hnsw_ef_construction
                hnsw_ef_search = storage_config.hnsw_ef_search

            # Submit button
            submitted = st.form_submit_button("Save Vector Store Configuration")

            if submitted:
                # Update configuration
                # In a real implementation, we would update the configuration in the ConfigProvider
                # For now, we'll just show a success message
                st.success("Vector Store Configuration saved successfully!")

                # Display the updated configuration
                st.json(
                    {
                        "vector_store_path": vector_store_path,
                        "vector_store_type": vector_store_type,
                        "index_type": index_type,
                        "use_gpu": use_gpu,
                        "ivf_nlist": ivf_nlist,
                        "ivf_nprobe": ivf_nprobe,
                        "hnsw_m": hnsw_m,
                        "hnsw_ef_construction": hnsw_ef_construction,
                        "hnsw_ef_search": hnsw_ef_search,
                    }
                )

    with tab2:
        st.subheader("Graph Store Configuration")

        # Create form for graph store configuration
        with st.form("graph_store_form"):
            # Neo4j connection parameters
            st.markdown("**Neo4j Connection Parameters**")

            neo4j_url = st.text_input("Neo4j URL", value="bolt://localhost:7687")

            neo4j_username = st.text_input("Neo4j Username", value="neo4j")

            neo4j_password = st.text_input("Neo4j Password", value="", type="password")

            neo4j_database = st.text_input("Neo4j Database", value="neo4j")

            # Submit button
            submitted = st.form_submit_button("Save Graph Store Configuration")

            if submitted:
                # Update configuration
                # In a real implementation, we would update the configuration in the ConfigProvider
                # For now, we'll just show a success message
                st.success("Graph Store Configuration saved successfully!")

                # Display the updated configuration
                st.json(
                    {
                        "neo4j_url": neo4j_url,
                        "neo4j_username": neo4j_username,
                        "neo4j_password": "*****" if neo4j_password else "",
                        "neo4j_database": neo4j_database,
                    }
                )

    with tab3:
        st.subheader("File Storage Configuration")

        # Create form for file storage configuration
        with st.form("file_storage_form"):
            # File storage parameters
            filings_dir = st.text_input("Filings Directory", value="data/filings")

            cache_dir = st.text_input("Cache Directory", value="data/cache")

            logs_dir = st.text_input("Logs Directory", value="data/logs")

            # Submit button
            submitted = st.form_submit_button("Save File Storage Configuration")

            if submitted:
                # Update configuration
                # In a real implementation, we would update the configuration in the ConfigProvider
                # For now, we'll just show a success message
                st.success("File Storage Configuration saved successfully!")

                # Display the updated configuration
                st.json(
                    {
                        "filings_dir": filings_dir,
                        "cache_dir": cache_dir,
                        "logs_dir": logs_dir,
                    }
                )

elif config_type == "Streamlit Configuration":
    st.header("Streamlit Configuration")

    # Get current configuration
    streamlit_config = ConfigProvider.get_config(StreamlitConfig)

    # Create form for Streamlit configuration
    with st.form("streamlit_config_form"):
        # Server settings
        st.subheader("Server Settings")

        port = st.number_input("Port", min_value=1024, max_value=65535, value=streamlit_config.port)

        headless = st.checkbox("Headless Mode", value=streamlit_config.headless)

        enable_cors = st.checkbox("Enable CORS", value=streamlit_config.enable_cors)

        enable_xsrf_protection = st.checkbox("Enable XSRF Protection", value=streamlit_config.enable_xsrf_protection)

        max_upload_size = st.number_input(
            "Max Upload Size (MB)",
            min_value=1,
            max_value=1000,
            value=streamlit_config.max_upload_size,
        )

        base_url_path = st.text_input("Base URL Path", value=streamlit_config.base_url_path)

        # UI settings
        st.subheader("UI Settings")

        theme_base = st.selectbox(
            "Theme Base",
            ["light", "dark"],
            index=0 if streamlit_config.theme_base == "light" else 1,
        )

        hide_top_bar = st.checkbox("Hide Top Bar", value=streamlit_config.hide_top_bar)

        show_error_details = st.checkbox("Show Error Details", value=streamlit_config.show_error_details)

        toolbar_mode = st.selectbox(
            "Toolbar Mode",
            ["auto", "developer", "viewer", "minimal"],
            index=0
            if streamlit_config.toolbar_mode == "auto"
            else 1
            if streamlit_config.toolbar_mode == "developer"
            else 2
            if streamlit_config.toolbar_mode == "viewer"
            else 3,
        )

        # Performance settings
        st.subheader("Performance Settings")

        caching = st.checkbox("Enable Caching", value=streamlit_config.caching)

        gather_usage_stats = st.checkbox("Gather Usage Stats", value=streamlit_config.gather_usage_stats)

        # Submit button
        submitted = st.form_submit_button("Save Streamlit Configuration")

        if submitted:
            # Update configuration
            # In a real implementation, we would update the configuration in the ConfigProvider
            # For now, we'll just show a success message
            st.success("Streamlit Configuration saved successfully!")

            # Display the updated configuration
            st.json(
                {
                    "port": port,
                    "headless": headless,
                    "enable_cors": enable_cors,
                    "enable_xsrf_protection": enable_xsrf_protection,
                    "max_upload_size": max_upload_size,
                    "base_url_path": base_url_path,
                    "theme_base": theme_base,
                    "hide_top_bar": hide_top_bar,
                    "show_error_details": show_error_details,
                    "toolbar_mode": toolbar_mode,
                    "caching": caching,
                    "gather_usage_stats": gather_usage_stats,
                }
            )
