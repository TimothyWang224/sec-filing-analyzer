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
class GraphStoreConfig:
    """Configuration for Graph Store settings."""
    
    max_cluster_size: int = int(os.getenv("GRAPH_MAX_CLUSTER_SIZE", "5"))

@dataclass
class ETLConfig:
    """Configuration for the ETL pipeline."""
    
    # Data retrieval settings
    cache_dir: Path = Path("data/cache/sec_filings")
    filing_types: List[str] = None
    max_retries: int = 3
    timeout: int = 30
    
    # Document processing settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"
    
    # Vector store settings
    vector_store_type: str = "simple"  # simple, pinecone, weaviate, etc.
    vector_store_path: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        # Set default filing types if not specified
        if self.filing_types is None:
            self.filing_types = ["10-K", "10-Q", "8-K"]
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set vector store path if not specified
        if self.vector_store_path is None:
            self.vector_store_path = Path("data/vector_store")
    
    @classmethod
    def from_env(cls) -> "ETLConfig":
        """Create configuration from environment variables."""
        return cls(
            cache_dir=Path(os.getenv("SEC_CACHE_DIR", "data/cache/sec_filings")),
            filing_types=os.getenv("SEC_FILING_TYPES", "10-K,10-Q,8-K").split(","),
            max_retries=int(os.getenv("SEC_MAX_RETRIES", "3")),
            timeout=int(os.getenv("SEC_TIMEOUT", "30")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            vector_store_type=os.getenv("VECTOR_STORE_TYPE", "simple"),
            vector_store_path=Path(os.getenv("VECTOR_STORE_PATH", "data/vector_store"))
        )

# Create global configuration instances
neo4j_config = Neo4jConfig()
graph_store_config = GraphStoreConfig()
etl_config = ETLConfig.from_env()

# Export configuration dictionaries for backward compatibility
NEO4J_CONFIG: Dict[str, Any] = {
    "username": neo4j_config.username,
    "password": neo4j_config.password,
    "url": neo4j_config.url,
    "database": neo4j_config.database,
}

GRAPH_STORE_CONFIG: Dict[str, Any] = {
    "max_cluster_size": graph_store_config.max_cluster_size,
} 