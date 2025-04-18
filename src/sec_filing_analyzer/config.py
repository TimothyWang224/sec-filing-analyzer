"""
Unified Configuration

Configuration settings for the SEC Filing Analyzer project.
"""

from typing import Dict, Any, List, Optional, Union, Type, TypeVar, Set
import os
import json
import glob
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class Neo4jConfig:
    """Configuration for Neo4j database connection."""

    username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    url: str = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")

@dataclass
class StorageConfig:
    """Configuration for storage settings."""

    # Graph store settings
    max_cluster_size: int = int(os.getenv("GRAPH_MAX_CLUSTER_SIZE", "5"))
    use_neo4j: bool = os.getenv("USE_NEO4J", "true").lower() == "true"
    graph_store_path: Optional[Path] = None

    # Vector store settings
    vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "simple")  # simple, pinecone, weaviate, etc.
    vector_store_path: Optional[Path] = None

    def __post_init__(self):
        """Initialize configuration after creation."""
        # Set vector store path if not specified
        if self.vector_store_path is None:
            self.vector_store_path = Path("data/vector_store")
            self.vector_store_path.mkdir(parents=True, exist_ok=True)

        # Set graph store path if not specified
        if self.graph_store_path is None:
            self.graph_store_path = Path("data/graph_store")
            self.graph_store_path.mkdir(parents=True, exist_ok=True)

@dataclass
class AgentConfig:
    """Configuration for agent parameters."""

    # Agent iteration parameters
    max_iterations: int = 3  # Legacy parameter, still used for backward compatibility
    max_planning_iterations: int = 2
    max_execution_iterations: int = 3
    max_refinement_iterations: int = 1

    # Tool execution parameters
    max_tool_retries: int = 2
    tools_per_iteration: int = 1  # Default to 1 for single tool call approach
    circuit_breaker_threshold: int = 3
    circuit_breaker_reset_timeout: int = 300

    # Runtime parameters
    max_duration_seconds: int = 180

    # Termination parameters
    enable_dynamic_termination: bool = False
    min_confidence_threshold: float = 0.8

    # LLM parameters
    llm_model: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4.1-nano")
    llm_temperature: float = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.7"))
    llm_max_tokens: int = int(os.getenv("DEFAULT_LLM_MAX_TOKENS", "4000"))

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables."""
        return cls(
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "3")),
            max_planning_iterations=int(os.getenv("AGENT_MAX_PLANNING_ITERATIONS", "2")),
            max_execution_iterations=int(os.getenv("AGENT_MAX_EXECUTION_ITERATIONS", "3")),
            max_refinement_iterations=int(os.getenv("AGENT_MAX_REFINEMENT_ITERATIONS", "1")),
            max_tool_retries=int(os.getenv("AGENT_MAX_TOOL_RETRIES", "2")),
            tools_per_iteration=int(os.getenv("AGENT_TOOLS_PER_ITERATION", "1")),
            circuit_breaker_threshold=int(os.getenv("AGENT_CIRCUIT_BREAKER_THRESHOLD", "3")),
            circuit_breaker_reset_timeout=int(os.getenv("AGENT_CIRCUIT_BREAKER_RESET_TIMEOUT", "300")),
            max_duration_seconds=int(os.getenv("AGENT_MAX_DURATION_SECONDS", "180")),
            enable_dynamic_termination=os.getenv("AGENT_ENABLE_DYNAMIC_TERMINATION", "false").lower() == "true",
            min_confidence_threshold=float(os.getenv("AGENT_MIN_CONFIDENCE_THRESHOLD", "0.8")),
            llm_model=os.getenv("DEFAULT_LLM_MODEL", "gpt-4.1-nano"),
            llm_temperature=float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("DEFAULT_LLM_MAX_TOKENS", "4000"))
        )

@dataclass
class ETLConfig:
    """Configuration for the ETL pipeline."""

    # Data retrieval settings
    filings_dir: Path = Path("data/filings")
    filing_types: List[str] = None
    max_retries: int = 2
    timeout: int = 30

    # Document processing settings
    chunk_size: int = 1024
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"

    # Parallel processing settings
    use_parallel: bool = True
    max_workers: int = 4
    batch_size: int = 100
    rate_limit: float = 0.1

    # XBRL extraction settings
    process_quantitative: bool = True
    db_path: str = "data/db_backup/improved_financial_data.duckdb"
    db_read_only: bool = True  # Default to read-only mode for database access

    # Processing flags
    process_semantic: bool = True
    delay_between_companies: int = 1

    def __post_init__(self):
        """Initialize configuration after creation."""
        # Set default filing types if not specified
        if self.filing_types is None:
            self.filing_types = ["10-K", "10-Q", "8-K"]

        # Create filings directory and subdirectories if they don't exist
        self.filings_dir.mkdir(parents=True, exist_ok=True)
        (self.filings_dir / "raw").mkdir(exist_ok=True)
        (self.filings_dir / "html").mkdir(exist_ok=True)
        (self.filings_dir / "xml").mkdir(exist_ok=True)
        (self.filings_dir / "processed").mkdir(exist_ok=True)
        (self.filings_dir / "cache").mkdir(exist_ok=True)

    @classmethod
    def from_env(cls) -> "ETLConfig":
        """Create configuration from environment variables."""
        return cls(
            filings_dir=Path(os.getenv("SEC_FILINGS_DIR", "data/filings")),
            filing_types=os.getenv("SEC_FILING_TYPES", "10-K,10-Q,8-K").split(","),
            max_retries=int(os.getenv("SEC_MAX_RETRIES", "3")),
            timeout=int(os.getenv("SEC_TIMEOUT", "30")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            use_parallel=os.getenv("USE_PARALLEL", "true").lower() == "true",
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            batch_size=int(os.getenv("BATCH_SIZE", "100")),
            rate_limit=float(os.getenv("RATE_LIMIT", "0.1")),
            process_semantic=os.getenv("PROCESS_SEMANTIC", "true").lower() == "true",
            process_quantitative=os.getenv("PROCESS_QUANTITATIVE", "true").lower() == "true",
            delay_between_companies=int(os.getenv("DELAY_BETWEEN_COMPANIES", "1"))
        )

@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""

    # Vector store type and path
    type: str = "optimized"  # optimized, flat, etc.
    path: Path = Path("data/vector_store")

    # Index parameters
    index_type: str = "hnsw"  # hnsw, flat, ivf, etc.
    use_gpu: bool = False

    # HNSW parameters
    hnsw_m: int = 32
    hnsw_ef_construction: int = 400
    hnsw_ef_search: int = 200

    # IVF parameters
    ivf_nlist: Optional[int] = None
    ivf_nprobe: Optional[int] = None

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        """Create configuration from environment variables."""
        return cls(
            type=os.getenv("VECTOR_STORE_TYPE", "optimized"),
            path=Path(os.getenv("VECTOR_STORE_PATH", "data/vector_store")),
            index_type=os.getenv("VECTOR_INDEX_TYPE", "hnsw"),
            use_gpu=os.getenv("VECTOR_USE_GPU", "false").lower() == "true",
            hnsw_m=int(os.getenv("VECTOR_HNSW_M", "32")),
            hnsw_ef_construction=int(os.getenv("VECTOR_HNSW_EF_CONSTRUCTION", "400")),
            hnsw_ef_search=int(os.getenv("VECTOR_HNSW_EF_SEARCH", "200")),
            ivf_nlist=int(os.getenv("VECTOR_IVF_NLIST", "0")) or None,
            ivf_nprobe=int(os.getenv("VECTOR_IVF_NPROBE", "0")) or None
        )

@dataclass
class StreamlitConfig:
    """Configuration for Streamlit applications."""

    # Server settings
    port: int = 8501
    headless: bool = True
    enable_cors: bool = True
    enable_xsrf_protection: bool = False
    max_upload_size: int = 200
    base_url_path: str = ""

    # UI settings
    theme_base: str = "light"
    hide_top_bar: bool = False
    show_error_details: bool = True
    toolbar_mode: str = "auto"

    # Performance settings
    caching: bool = False
    gather_usage_stats: bool = False

    @classmethod
    def from_env(cls) -> "StreamlitConfig":
        """Create configuration from environment variables."""
        return cls(
            port=int(os.getenv("STREAMLIT_PORT", "8501")),
            headless=os.getenv("STREAMLIT_HEADLESS", "true").lower() == "true",
            enable_cors=os.getenv("STREAMLIT_ENABLE_CORS", "true").lower() == "true",
            enable_xsrf_protection=os.getenv("STREAMLIT_ENABLE_XSRF_PROTECTION", "false").lower() == "true",
            max_upload_size=int(os.getenv("STREAMLIT_MAX_UPLOAD_SIZE", "200")),
            base_url_path=os.getenv("STREAMLIT_BASE_URL_PATH", ""),
            theme_base=os.getenv("STREAMLIT_THEME_BASE", "light"),
            hide_top_bar=os.getenv("STREAMLIT_HIDE_TOP_BAR", "false").lower() == "true",
            show_error_details=os.getenv("STREAMLIT_SHOW_ERROR_DETAILS", "true").lower() == "true",
            toolbar_mode=os.getenv("STREAMLIT_TOOLBAR_MODE", "auto"),
            caching=os.getenv("STREAMLIT_CACHING", "false").lower() == "true",
            gather_usage_stats=os.getenv("STREAMLIT_GATHER_USAGE_STATS", "false").lower() == "true"
        )

# Type variable for configuration classes
T = TypeVar('T')

class ConfigProvider:
    """Unified configuration provider for the SEC Filing Analyzer."""

    # Configuration instances
    _neo4j_config: Optional[Neo4jConfig] = None
    _storage_config: Optional[StorageConfig] = None
    _etl_config: Optional[ETLConfig] = None
    _agent_config: Optional[AgentConfig] = None
    _vector_store_config: Optional[VectorStoreConfig] = None
    _streamlit_config: Optional[StreamlitConfig] = None

    # Agent-specific configurations from llm_config.py
    _agent_specific_configs: Dict[str, Dict[str, Any]] = {}

    # Tool schemas
    _tool_schemas: Dict[str, Dict[str, Any]] = {}

    # External configuration file path
    _external_config_path: Optional[Path] = None

    # Schema directory path
    _schema_dir: Optional[Path] = None

    @classmethod
    def initialize(cls, config_path: Optional[str] = None, schema_dir: Optional[str] = None) -> None:
        """Initialize the configuration provider."""
        # Set external config path if provided
        if config_path:
            cls._external_config_path = Path(config_path)
        else:
            # Default to data/config/etl_config.json
            cls._external_config_path = Path("data/config/etl_config.json")

        # Set schema directory path if provided
        if schema_dir:
            cls._schema_dir = Path(schema_dir)
        else:
            # Default to data/schemas
            cls._schema_dir = Path("data/schemas")

        # Initialize configuration instances
        cls._neo4j_config = Neo4jConfig()
        cls._storage_config = StorageConfig()
        cls._etl_config = ETLConfig.from_env()
        cls._agent_config = AgentConfig.from_env()
        cls._vector_store_config = VectorStoreConfig.from_env()
        cls._streamlit_config = StreamlitConfig.from_env()

        # Load agent-specific configurations from llm_config.py
        try:
            from sec_filing_analyzer.llm.llm_config import AGENT_CONFIGS
            cls._agent_specific_configs = AGENT_CONFIGS
        except ImportError:
            logger.warning("Could not import agent-specific configurations from llm_config.py")

        # Load tool schemas
        cls._load_tool_schemas()

    @classmethod
    def get_config(cls, config_type: Type[T]) -> T:
        """Get a configuration instance by type."""
        if not cls._neo4j_config:
            cls.initialize()

        if config_type == Neo4jConfig:
            return cls._neo4j_config  # type: ignore
        elif config_type == StorageConfig:
            return cls._storage_config  # type: ignore
        elif config_type == ETLConfig:
            return cls._etl_config  # type: ignore
        elif config_type == AgentConfig:
            return cls._agent_config  # type: ignore
        elif config_type == VectorStoreConfig:
            return cls._vector_store_config  # type: ignore
        elif config_type == StreamlitConfig:
            return cls._streamlit_config  # type: ignore
        else:
            raise ValueError(f"Unknown configuration type: {config_type}")

    @classmethod
    def get_agent_config(cls, agent_type: str) -> Dict[str, Any]:
        """Get configuration for a specific agent type."""
        if not cls._agent_config:
            cls.initialize()

        # Start with base agent config
        config = asdict(cls._agent_config)

        # Apply agent-specific config if available
        if agent_type in cls._agent_specific_configs:
            agent_specific = cls._agent_specific_configs[agent_type]

            # Update with agent-specific values
            for key, value in agent_specific.items():
                config[key] = value

        # Apply external config if available
        if cls._external_config_path and cls._external_config_path.exists():
            try:
                with open(cls._external_config_path, 'r') as f:
                    external_config = json.load(f)

                if 'agent' in external_config:
                    # Update with external values
                    for key, value in external_config['agent'].items():
                        config[key] = value
            except Exception as e:
                print(f"Warning: Could not load external config: {str(e)}")

        return config

    @classmethod
    def get_all_agent_types(cls) -> List[str]:
        """Get all available agent types."""
        if not cls._agent_specific_configs:
            cls.initialize()

        return list(cls._agent_specific_configs.keys())

    @classmethod
    def get_tool_schema(cls, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool."""
        if not cls._tool_schemas:
            cls._load_tool_schemas()

        return cls._tool_schemas.get(tool_name)

    @classmethod
    def get_all_tool_schemas(cls) -> Dict[str, Dict[str, Any]]:
        """Get all tool schemas."""
        if not cls._tool_schemas:
            cls._load_tool_schemas()

        return cls._tool_schemas.copy()

    @classmethod
    def _load_tool_schemas(cls) -> None:
        """Load tool parameter schemas from JSON files."""
        if not cls._schema_dir:
            cls._schema_dir = Path("data/schemas")

        # Create schema directory if it doesn't exist
        if not cls._schema_dir.exists():
            cls._schema_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created schema directory: {cls._schema_dir}")

        # Load schemas from JSON files
        schemas = {}
        for schema_file in cls._schema_dir.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                    tool_name = schema_file.stem
                    schemas[tool_name] = schema
                    logger.debug(f"Loaded schema for tool: {tool_name}")
            except Exception as e:
                logger.warning(f"Error loading schema from {schema_file}: {str(e)}")

        # If no schemas were found, try to load from the tool_parameter_helper.py file
        if not schemas:
            try:
                from src.tools.tool_parameter_helper import TOOL_PARAMETER_SCHEMAS
                schemas = TOOL_PARAMETER_SCHEMAS
                logger.info("Loaded schemas from tool_parameter_helper.py")

                # Save schemas to JSON files for future use
                for tool_name, schema in schemas.items():
                    schema_file = cls._schema_dir / f"{tool_name}.json"
                    try:
                        with open(schema_file, 'w') as f:
                            json.dump(schema, f, indent=2)
                        logger.debug(f"Saved schema for tool: {tool_name}")
                    except Exception as e:
                        logger.warning(f"Error saving schema to {schema_file}: {str(e)}")
            except ImportError:
                logger.warning("Could not import TOOL_PARAMETER_SCHEMAS from tool_parameter_helper.py")

        cls._tool_schemas = schemas

    @classmethod
    def validate_config(cls, config: Dict[str, Any], config_type: str) -> bool:
        """Validate a configuration dictionary."""
        if config_type == 'agent':
            required_fields = {
                'max_iterations', 'max_planning_iterations', 'max_execution_iterations',
                'max_refinement_iterations', 'llm_model', 'llm_temperature', 'llm_max_tokens'
            }
            return all(field in config for field in required_fields)
        elif config_type == 'etl':
            required_fields = {'filings_dir', 'filing_types', 'max_retries', 'timeout'}
            return all(field in config for field in required_fields)
        elif config_type == 'vector_store':
            required_fields = {'type', 'path', 'index_type'}
            return all(field in config for field in required_fields)
        elif config_type == 'streamlit':
            required_fields = {'port', 'headless', 'enable_cors'}
            return all(field in config for field in required_fields)
        else:
            return True  # No validation for other types yet

# Create global configuration instances
neo4j_config = Neo4jConfig()
storage_config = StorageConfig()
etl_config = ETLConfig.from_env()
agent_config = AgentConfig.from_env()
vector_store_config = VectorStoreConfig.from_env()
streamlit_config = StreamlitConfig.from_env()

# Initialize the configuration provider
ConfigProvider.initialize()

# Export configuration dictionaries for backward compatibility
NEO4J_CONFIG: Dict[str, Any] = {
    "username": neo4j_config.username,
    "password": neo4j_config.password,
    "url": neo4j_config.url,
    "database": neo4j_config.database,
}

STORAGE_CONFIG: Dict[str, Any] = {
    "max_cluster_size": storage_config.max_cluster_size,
    "use_neo4j": storage_config.use_neo4j,
    "vector_store_type": storage_config.vector_store_type,
    "vector_store_path": str(storage_config.vector_store_path),
    "graph_store_path": str(storage_config.graph_store_path)
}

# Export agent configuration dictionary for backward compatibility
AGENT_CONFIG: Dict[str, Any] = {
    # Agent iteration parameters
    "max_iterations": agent_config.max_iterations,
    "max_planning_iterations": agent_config.max_planning_iterations,
    "max_execution_iterations": agent_config.max_execution_iterations,
    "max_refinement_iterations": agent_config.max_refinement_iterations,

    # Tool execution parameters
    "max_tool_retries": agent_config.max_tool_retries,
    "tools_per_iteration": agent_config.tools_per_iteration,
    "circuit_breaker_threshold": agent_config.circuit_breaker_threshold,
    "circuit_breaker_reset_timeout": agent_config.circuit_breaker_reset_timeout,

    # Runtime parameters
    "max_duration_seconds": agent_config.max_duration_seconds,

    # Termination parameters
    "enable_dynamic_termination": agent_config.enable_dynamic_termination,
    "min_confidence_threshold": agent_config.min_confidence_threshold,

    # LLM parameters
    "llm_model": agent_config.llm_model,
    "llm_temperature": agent_config.llm_temperature,
    "llm_max_tokens": agent_config.llm_max_tokens
}