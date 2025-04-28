"""
Test suite for SEC filing data retrieval components.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from sec_filing_analyzer.data_retrieval.file_storage import FileStorage
from sec_filing_analyzer.data_retrieval.filing_processor import FilingProcessor
from sec_filing_analyzer.data_retrieval.sec_downloader import SECFilingsDownloader
from sec_filing_analyzer.storage import GraphStore, LlamaIndexVectorStore

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

SAMPLE_CHUNKS = pd.DataFrame({"text": ["Chunk 1", "Chunk 2", "Chunk 3"], "start": [0, 100, 200], "end": [99, 199, 299]})


@pytest.fixture
def sample_filing_data():
    """Return sample filing data."""
    return SAMPLE_FILING_DATA.copy()

@pytest.fixture
def sample_chunks():
    """Return sample chunks data."""
    return SAMPLE_CHUNKS.copy()

@pytest.fixture
def mock_sec_downloader():
    """Create a mock SEC downloader."""
    with patch("sec_filing_analyzer.data_retrieval.sec_downloader.edgar") as mock_edgar:
        mock_company = Mock()
        mock_company.get_filings.return_value = [SAMPLE_FILING_DATA]
        mock_edgar.Company.return_value = mock_company
        yield mock_edgar


@pytest.fixture
def mock_file_storage(tmp_path):
    """Create a mock file storage."""
    storage = FileStorage(base_dir=tmp_path)
    return storage


@pytest.fixture
def mock_graph_store():
    """Create a mock graph store."""
    return Mock(spec=GraphStore)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    return Mock(spec=LlamaIndexVectorStore)


@pytest.fixture
def filing_processor(mock_graph_store, mock_vector_store, mock_file_storage):
    """Create a filing processor with mocked dependencies."""
    return FilingProcessor(graph_store=mock_graph_store, vector_store=mock_vector_store, file_storage=mock_file_storage)


class TestSECFilingsDownloader:
    """Test cases for SEC downloader."""

    def test_download_filing(self, sample_filing_data):
        """Test downloading a filing."""
        # Create a mock Filing object
        mock_filing = Mock()
        mock_filing.accession_number = sample_filing_data["accession_number"]
        mock_filing.form = sample_filing_data["form"]
        mock_filing.text.return_value = sample_filing_data["text"]

        # Create a mock file_storage
        mock_storage = Mock()
        mock_storage.load_cached_filing.return_value = None  # No cached data

        # Setup the downloader with the mock storage
        downloader = SECFilingsDownloader(file_storage=mock_storage)

        # Patch the edgar_utils.get_filing_metadata function
        with patch("sec_filing_analyzer.utils.edgar_utils.get_filing_metadata") as mock_get_metadata:
            # Return the sample filing data as metadata
            metadata = {
                "accession_number": sample_filing_data["accession_number"],
                "form": sample_filing_data["form"],
                "id": sample_filing_data["accession_number"],
                "filing_type": sample_filing_data["form"],
                "company": sample_filing_data["company"],
                "ticker": "AAPL"
            }
            mock_get_metadata.return_value = metadata

            # Patch the edgar_utils.get_filing_content function
            with patch("sec_filing_analyzer.utils.edgar_utils.get_filing_content") as mock_get_content:
                mock_get_content.return_value = {
                    "text": sample_filing_data["text"],
                    "html": None,
                    "xml": None
                }

                # Call the method with the mock filing
                filing_metadata = downloader.download_filing(mock_filing, "AAPL")

                assert filing_metadata is not None
                assert filing_metadata["accession_number"] == sample_filing_data["accession_number"]
                assert filing_metadata["form"] == sample_filing_data["form"]

                # Verify that save_raw_filing was called
                mock_storage.save_raw_filing.assert_called_once()

    def test_download_filing_html(self, sample_filing_data):
        """Test downloading filing HTML content."""
        # Create a mock Filing object
        mock_filing = Mock()
        mock_filing.accession_number = sample_filing_data["accession_number"]
        mock_filing.form = sample_filing_data["form"]
        mock_filing.html.return_value = sample_filing_data["html_content"]

        # Setup the downloader with a mock file_storage
        mock_storage = Mock()
        mock_storage.load_cached_filing.return_value = None  # No cached data
        downloader = SECFilingsDownloader(file_storage=mock_storage)

        # Patch the edgar_utils.get_filing_metadata function
        with patch("sec_filing_analyzer.utils.edgar_utils.get_filing_metadata") as mock_get_metadata:
            # Return the sample filing data as metadata
            metadata = {
                "accession_number": sample_filing_data["accession_number"],
                "form": sample_filing_data["form"],
                "id": sample_filing_data["accession_number"],
                "filing_type": sample_filing_data["form"],
                "company": sample_filing_data["company"],
                "ticker": "AAPL"
            }
            mock_get_metadata.return_value = metadata

            # Patch the edgar_utils.get_filing_content function
            with patch("sec_filing_analyzer.utils.edgar_utils.get_filing_content") as mock_get_content:
                mock_get_content.return_value = {
                    "text": sample_filing_data["text"],
                    "html": sample_filing_data["html_content"],
                    "xml": None
                }

                # Call the method with the mock filing
                filing_metadata = downloader.download_filing(mock_filing, "AAPL")

                assert filing_metadata is not None
                # Verify that save_html_filing was called with the HTML content
                mock_storage.save_html_filing.assert_called_once()

    @pytest.mark.parametrize(
        "filing_type,start_date,end_date",
        [
            ("10-K", "2023-01-01", "2023-12-31"),
            ("10-Q", "2023-01-01", "2023-12-31"),
            ("8-K", "2023-01-01", "2023-12-31"),
        ],
    )
    def test_download_filing_types(self, filing_type, start_date, end_date):
        """Test downloading different filing types."""
        # Setup the downloader
        downloader = SECFilingsDownloader()

        # Patch the edgar_utils.get_filings function
        with patch("sec_filing_analyzer.utils.edgar_utils.get_filings") as mock_get_filings:
            # Create a mock Filing object
            mock_filing = Mock()
            mock_filing.accession_number = "test-accession"
            mock_filing.form = filing_type
            mock_filing.text.return_value = "Sample text"

            # Return the mock filing
            mock_get_filings.return_value = [mock_filing]

            # Patch the download_filing method
            with patch.object(downloader, "download_filing") as mock_download:
                mock_download.return_value = {
                    "accession_number": "test-accession",
                    "form": filing_type
                }

                # Call the method
                filings = downloader.download_company_filings(
                    ticker="AAPL",
                    filing_types=[filing_type],
                    start_date=start_date,
                    end_date=end_date,
                    limit=1
                )

                assert len(filings) > 0
                assert filings[0]["form"] == filing_type


class TestFilingProcessor:
    """Test cases for filing processor."""

    def test_process_filing(self, filing_processor, sample_filing_data):
        """Test processing a filing."""
        # Add required fields to sample_filing_data
        filing_data = sample_filing_data.copy()
        filing_data["id"] = filing_data["accession_number"]
        filing_data["embedding"] = [0.1] * 10  # Mock embedding
        filing_data["metadata"] = {
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2023-01-01",
            "company": "Apple Inc."
        }

        # Mock the vector store and graph store
        filing_processor.vector_store.upsert_vectors.return_value = True
        filing_processor.graph_store.add_filing.return_value = True

        # Mock the file_storage methods
        mock_cache_filing = Mock(return_value=True)
        mock_load_cached = Mock(return_value=None)

        # Replace the methods on the filing_processor.file_storage
        filing_processor.file_storage.cache_filing = mock_cache_filing
        filing_processor.file_storage.load_cached_filing = mock_load_cached

        processed_data = filing_processor.process_filing(filing_data)

        assert processed_data is not None
        assert "id" in processed_data
        assert "text" in processed_data
        assert "metadata" in processed_data

        # Verify graph store interactions
        filing_processor.graph_store.add_filing.assert_called_once()

        # Verify vector store interactions
        filing_processor.vector_store.upsert_vectors.assert_called()

        # Verify file_storage interactions
        mock_cache_filing.assert_called_once()

    def test_process_filing_with_chunks(self, filing_processor, sample_filing_data, sample_chunks):
        """Test processing a filing with chunks."""
        # Add required fields to sample_filing_data
        filing_data = sample_filing_data.copy()
        filing_data["id"] = filing_data["accession_number"]
        filing_data["embedding"] = [0.1] * 10  # Mock embedding
        filing_data["metadata"] = {
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2023-01-01",
            "company": "Apple Inc."
        }
        filing_data["chunks"] = sample_chunks.to_dict("records")
        filing_data["chunk_embeddings"] = [[0.1] * 10 for _ in range(len(sample_chunks))]  # Mock chunk embeddings
        filing_data["chunk_texts"] = [chunk["text"] for chunk in filing_data["chunks"]]

        # Mock the vector store and graph store
        filing_processor.vector_store.upsert_vectors.return_value = True
        filing_processor.graph_store.add_filing.return_value = True

        # Mock the file_storage methods
        mock_cache_filing = Mock(return_value=True)
        mock_load_cached = Mock(return_value=None)

        # Replace the methods on the filing_processor.file_storage
        filing_processor.file_storage.cache_filing = mock_cache_filing
        filing_processor.file_storage.load_cached_filing = mock_load_cached

        processed_data = filing_processor.process_filing(filing_data)

        assert processed_data is not None
        assert "chunks" in processed_data
        assert len(processed_data["chunks"]) == len(sample_chunks)

        # Verify file_storage interactions
        mock_cache_filing.assert_called_once()

    def test_get_filing(self, filing_processor, sample_filing_data):
        """Test retrieving a filing."""
        filing_id = sample_filing_data["accession_number"]

        # Create a mock for the file_storage methods
        mock_load_cached = Mock(return_value={"processed_data": sample_filing_data})
        mock_load_processed = Mock(return_value=sample_filing_data)
        mock_load_raw = Mock(return_value=None)

        # Replace the methods on the filing_processor.file_storage
        filing_processor.file_storage.load_cached_filing = mock_load_cached
        filing_processor.file_storage.load_processed_filing = mock_load_processed
        filing_processor.file_storage.load_raw_filing = mock_load_raw

        # Test cached data
        filing = filing_processor.get_filing(filing_id)
        assert filing is not None
        assert filing["processed_data"]["accession_number"] == filing_id

        # Test processed data
        mock_load_cached.return_value = None
        filing = filing_processor.get_filing(filing_id)
        assert filing is not None
        assert filing["accession_number"] == filing_id


class TestFileStorage:
    """Test cases for file storage."""

    def test_save_raw_filing(self, temp_storage_dir, sample_filing_data):
        """Test saving raw filing data."""
        # Create a real FileStorage instance with the temp directory
        file_storage = FileStorage(base_dir=temp_storage_dir)

        # Add required fields to sample_filing_data
        filing_data = sample_filing_data.copy()
        filing_data["ticker"] = "AAPL"
        filing_data["filing_date"] = "2023-01-01"

        filing_id = filing_data["accession_number"]

        # Save the raw filing
        file_storage.save_raw_filing(
            filing_id=filing_id, content=filing_data["text"], metadata=filing_data
        )

        # Verify file was created - note the path structure includes ticker and year
        raw_file = temp_storage_dir / "raw" / "AAPL" / "2023" / f"{filing_id}.txt"
        assert raw_file.exists()

    def test_save_html_filing(self, temp_storage_dir, sample_filing_data):
        """Test saving HTML filing data."""
        # Create a real FileStorage instance with the temp directory
        file_storage = FileStorage(base_dir=temp_storage_dir)

        # Add required fields to sample_filing_data
        filing_data = sample_filing_data.copy()
        filing_data["ticker"] = "AAPL"
        filing_data["filing_date"] = "2023-01-01"

        filing_id = filing_data["accession_number"]

        # Save the HTML filing
        file_storage.save_html_filing(
            filing_id=filing_id, html_content=filing_data["html_content"], metadata=filing_data
        )

        # Verify file was created - note the path structure includes ticker and year
        html_file = temp_storage_dir / "html" / "AAPL" / "2023" / f"{filing_id}.html"
        assert html_file.exists()

    def test_save_processed_filing(self, temp_storage_dir, sample_filing_data, sample_chunks):
        """Test saving processed filing data."""
        # Create a real FileStorage instance with the temp directory
        file_storage = FileStorage(base_dir=temp_storage_dir)

        # Add required fields to sample_filing_data
        filing_data = sample_filing_data.copy()
        filing_data["ticker"] = "AAPL"
        filing_data["filing_date"] = "2023-01-01"

        filing_id = filing_data["accession_number"]

        processed_data = {
            "filing_id": filing_id,
            "text": filing_data["text"],
            "metadata": filing_data,
            "chunks": sample_chunks.to_dict("records"),
        }

        # Save the processed filing
        file_storage.save_processed_filing(
            filing_id=filing_id, processed_data=processed_data, metadata=filing_data
        )

        # Verify file was created - note the path structure includes ticker and year
        processed_file = temp_storage_dir / "processed" / "AAPL" / "2023" / f"{filing_id}_processed.json"
        assert processed_file.exists()

    def test_list_filings(self, temp_storage_dir):
        """Test listing available filings."""
        # Create a real FileStorage instance with the temp directory
        file_storage = FileStorage(base_dir=temp_storage_dir)

        # Test filtering by ticker
        filings = file_storage.list_filings(ticker="AAPL")
        assert isinstance(filings, list)

        # Test filtering by year
        filings = file_storage.list_filings(year="2023")
        assert isinstance(filings, list)

        # Test filtering by filing type
        filings = file_storage.list_filings(filing_type="10-K")
        assert isinstance(filings, list)
