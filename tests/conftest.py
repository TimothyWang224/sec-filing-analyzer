"""
Shared test fixtures and data for SEC filing analyzer tests.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Import environment if available, otherwise use Mock
try:
    from sec_filing_analyzer.environment import FinancialEnvironment
except ImportError:
    FinancialEnvironment = Mock

# Test data
SAMPLE_FILING_DATA = {
    "accession_number": "0000320193-23-000077",
    "form": "10-K",
    "filing_date": "2023-10-27",
    "company": "Apple Inc.",
    "ticker": "AAPL",
    "description": "Annual report",
    "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019323000077/aapl-20230930.htm",
    "text": "Sample filing text content...",
    "html_content": "<html><body>Sample HTML content</body></html>",
}

SAMPLE_CHUNKS = pd.DataFrame(
    {
        "text": ["Chunk 1", "Chunk 2", "Chunk 3"],
        "start": [0, 100, 200],
        "end": [99, 199, 299],
    }
)

SAMPLE_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def sample_filing_data():
    """Provide sample filing data."""
    return SAMPLE_FILING_DATA.copy()


@pytest.fixture
def sample_chunks():
    """Provide sample chunks data."""
    return SAMPLE_CHUNKS.copy()


@pytest.fixture
def sample_embedding():
    """Provide sample embedding data."""
    return SAMPLE_EMBEDDING.copy()


@pytest.fixture
def mock_edgar():
    """Create a mock edgar library."""
    with patch("sec_filing_analyzer.data_retrieval.sec_downloader.edgar") as mock_edgar:
        mock_company = Mock()
        mock_company.get_filings.return_value = [SAMPLE_FILING_DATA]
        mock_edgar.Company.return_value = mock_company
        yield mock_edgar


@pytest.fixture
def mock_openai_embedding():
    """Create a mock OpenAI embedding model."""
    with patch(
        "llama_index.embeddings.openai.OpenAIEmbedding"
    ) as mock_embedding:
        mock_model = Mock()
        mock_model.get_text_embedding.return_value = SAMPLE_EMBEDDING
        mock_embedding.return_value = mock_model
        yield mock_embedding


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()

    # Create subdirectories
    (storage_dir / "raw").mkdir()
    (storage_dir / "html").mkdir()
    (storage_dir / "processed").mkdir()
    (storage_dir / "cache").mkdir()

    return storage_dir


@pytest.fixture
def mock_graph_store():
    """Create a mock graph store."""
    from sec_filing_analyzer.storage import GraphStore

    return Mock(spec=GraphStore)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    from sec_filing_analyzer.storage import LlamaIndexVectorStore

    return Mock(spec=LlamaIndexVectorStore)


@pytest.fixture
def mock_filing_processor():
    """Create a mock filing processor."""
    from sec_filing_analyzer.data_retrieval.filing_processor import FilingProcessor

    return Mock(spec=FilingProcessor)


@pytest.fixture
def mock_file_storage():
    """Create a mock file storage."""
    from sec_filing_analyzer.data_retrieval.file_storage import FileStorage

    return Mock(spec=FileStorage)


@pytest.fixture
def etl_pipeline(
    mock_graph_store, mock_vector_store, mock_filing_processor, mock_file_storage
):
    """Create an ETL pipeline with mocked dependencies."""
    from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline

    return SECFilingETLPipeline(
        graph_store=mock_graph_store,
        vector_store=mock_vector_store,
        filing_processor=mock_filing_processor,
        file_storage=mock_file_storage,
    )


# Additional fixtures for script tests


@pytest.fixture
def question() -> str:
    """Provide a sample question for QA tests."""
    return "What was MSFT's net income for 2023?"


@pytest.fixture
def ticker() -> str:
    """Provide a sample ticker for financial data tests."""
    return "AAPL"


@pytest.fixture
def metric() -> str:
    """Provide a sample metric for financial data tests."""
    return "Revenue"


@pytest.fixture
def accession_number() -> str:
    """Provide a sample accession number for filing tests."""
    return "0000320193-23-000077"


@pytest.fixture
def query() -> str:
    """Provide a sample query for search tests."""
    return "What are the risk factors mentioned in the latest 10-K?"


@pytest.fixture
def companies() -> list:
    """Provide a sample list of companies."""
    return ["AAPL", "MSFT", "GOOG", "NVDA"]


@pytest.fixture
def filing_type() -> str:
    """Provide a sample filing type."""
    return "10-K"


@pytest.fixture
def year() -> int:
    """Provide a sample year."""
    return 2023


@pytest.fixture
def quarter() -> int:
    """Provide a sample quarter."""
    return 4


@pytest.fixture
def date_range() -> tuple:
    """Provide a sample date range."""
    return ("2023-01-01", "2023-12-31")


@pytest.fixture
def env():
    """Provide a mock financial environment."""
    try:
        return FinancialEnvironment()
    except:
        return Mock()


@pytest.fixture
def tool_name() -> str:
    """Provide a sample tool name."""
    return "SECFinancialDataTool"


@pytest.fixture
def parameter_name() -> str:
    """Provide a sample parameter name."""
    return "ticker"


@pytest.fixture
def parameter_value() -> str:
    """Provide a sample parameter value."""
    return "AAPL"
