"""
Update Unified Configuration

This script updates the unified configuration file with all the settings from various sources.
"""

import sys
import json
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    # Import the ConfigProvider
    from src.sec_filing_analyzer.config import ConfigProvider, VectorStoreConfig, StreamlitConfig, ETLConfig
    
    # Initialize the ConfigProvider
    ConfigProvider.initialize()
    
    # Load the unified configuration file
    config_path = Path("data/config/etl_config.json")
    if not config_path.exists():
        print(f"Unified configuration file not found: {config_path}")
        print("Creating a new configuration file...")
        config = {}
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Update the configuration with the VectorStoreConfig
    vector_store_config = ConfigProvider.get_config(VectorStoreConfig)
    config["vector_store"] = {
        "type": vector_store_config.type,
        "path": str(vector_store_config.path),
        "index_type": vector_store_config.index_type,
        "use_gpu": vector_store_config.use_gpu,
        "hnsw_m": vector_store_config.hnsw_m,
        "hnsw_ef_construction": vector_store_config.hnsw_ef_construction,
        "hnsw_ef_search": vector_store_config.hnsw_ef_search,
        "ivf_nlist": vector_store_config.ivf_nlist,
        "ivf_nprobe": vector_store_config.ivf_nprobe
    }
    
    # Update the configuration with the StreamlitConfig
    streamlit_config = ConfigProvider.get_config(StreamlitConfig)
    config["streamlit"] = {
        "server": {
            "port": streamlit_config.port,
            "headless": streamlit_config.headless,
            "enable_cors": streamlit_config.enable_cors,
            "enable_xsrf_protection": streamlit_config.enable_xsrf_protection,
            "max_upload_size": streamlit_config.max_upload_size,
            "base_url_path": streamlit_config.base_url_path
        },
        "theme": {
            "base": streamlit_config.theme_base
        },
        "ui": {
            "hide_top_bar": streamlit_config.hide_top_bar
        },
        "client": {
            "show_error_details": streamlit_config.show_error_details,
            "toolbar_mode": streamlit_config.toolbar_mode,
            "caching": streamlit_config.caching
        }
    }
    
    # Update the configuration with the ETLConfig
    etl_config = ConfigProvider.get_config(ETLConfig)
    config["etl_pipeline"] = {
        "filings_dir": str(etl_config.filings_dir),
        "filing_types": etl_config.filing_types,
        "max_retries": etl_config.max_retries,
        "timeout": etl_config.timeout,
        "chunk_size": etl_config.chunk_size,
        "chunk_overlap": etl_config.chunk_overlap,
        "embedding_model": etl_config.embedding_model,
        "use_parallel": etl_config.use_parallel,
        "max_workers": etl_config.max_workers,
        "batch_size": etl_config.batch_size,
        "rate_limit": etl_config.rate_limit,
        "process_quantitative": etl_config.process_quantitative,
        "process_semantic": etl_config.process_semantic,
        "delay_between_companies": etl_config.delay_between_companies
    }
    
    # Save the updated configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Updated unified configuration file: {config_path}")
    
except Exception as e:
    print(f"Error updating unified configuration: {str(e)}")
    import traceback
    traceback.print_exc()
