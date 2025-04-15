"""
Unified Configuration

Configuration settings for the SEC Filing Analyzer project.
"""

from typing import Dict, Any, List, Optional
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

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
    llm_model: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
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
            llm_model=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("DEFAULT_LLM_MAX_TOKENS", "4000"))
        )

@dataclass
class ETLConfig:
    """Configuration for the ETL pipeline."""

    # Data retrieval settings
    filings_dir: Path = Path("data/filings")
    filing_types: List[str] = None
    max_retries: int = 3
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
    db_path: str = "data/financial_data.duckdb"

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
            rate_limit=float(os.getenv("RATE_LIMIT", "0.1"))
        )

# Create global configuration instances
neo4j_config = Neo4jConfig()
storage_config = StorageConfig()
etl_config = ETLConfig.from_env()
agent_config = AgentConfig.from_env()

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