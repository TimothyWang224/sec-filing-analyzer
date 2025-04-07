"""
Test suite for SEC filing data retrieval components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
from datetime import datetime

from sec_filing_analyzer.data_retrieval.sec_downloader import SECDownloader
from sec_filing_analyzer.data_retrieval.filing_processor import FilingProcessor
from sec_filing_analyzer.data_retrieval.file_storage import FileStorage
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
    "html_content": "<html><body>Sample HTML content</body></html>"
}

SAMPLE_CHUNKS = pd.DataFrame({
    'text': ['Chunk 1', 'Chunk 2', 'Chunk 3'],
    'start': [0, 100, 200],
    'end': [99, 199, 299]
})

@pytest.fixture
def mock_sec_downloader():
    """Create a mock SEC downloader."""
    with patch('sec_filing_analyzer.data_retrieval.sec_downloader.edgar') as mock_edgar:
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
    return FilingProcessor(
        graph_store=mock_graph_store,
        vector_store=mock_vector_store,
        file_storage=mock_file_storage
    )

class TestSECDownloader:
    """Test cases for SEC downloader."""
    
    def test_download_filing(self, mock_edgar, sample_filing_data):
        """Test downloading a filing."""
        downloader = SECDownloader()
        filing = downloader.download_filing("AAPL", "10-K", "2023-01-01", "2023-12-31")
        
        assert filing is not None
        assert filing["accession_number"] == sample_filing_data["accession_number"]
        assert filing["form"] == sample_filing_data["form"]
    
    def test_download_filing_html(self, mock_edgar, sample_filing_data):
        """Test downloading filing HTML content."""
        downloader = SECDownloader()
        html_content = downloader.download_filing_html(sample_filing_data)
        
        assert html_content is not None
        assert isinstance(html_content, str)
    
    @pytest.mark.parametrize("filing_type,start_date,end_date", [
        ("10-K", "2023-01-01", "2023-12-31"),
        ("10-Q", "2023-01-01", "2023-12-31"),
        ("8-K", "2023-01-01", "2023-12-31")
    ])
    def test_download_filing_types(self, mock_edgar, filing_type, start_date, end_date):
        """Test downloading different filing types."""
        downloader = SECDownloader()
        filing = downloader.download_filing("AAPL", filing_type, start_date, end_date)
        
        assert filing is not None
        assert filing["form"] == filing_type

class TestFilingProcessor:
    """Test cases for filing processor."""
    
    def test_process_filing(self, filing_processor, sample_filing_data):
        """Test processing a filing."""
        processed_data = filing_processor.process_filing(sample_filing_data)
        
        assert processed_data is not None
        assert "filing_id" in processed_data
        assert "text" in processed_data
        assert "metadata" in processed_data
        
        # Verify graph store interactions
        filing_processor.graph_store.add_filing.assert_called_once()
        
        # Verify vector store interactions
        filing_processor.vector_store.upsert_vectors.assert_called()
    
    def test_process_filing_with_chunks(self, filing_processor, sample_filing_data, sample_chunks):
        """Test processing a filing with chunks."""
        filing_data = sample_filing_data.copy()
        filing_data["chunks"] = sample_chunks.to_dict('records')
        
        processed_data = filing_processor.process_filing(filing_data)
        
        assert processed_data is not None
        assert "chunks" in processed_data
        assert len(processed_data["chunks"]) == len(sample_chunks)
    
    def test_get_filing(self, filing_processor, mock_file_storage, sample_filing_data):
        """Test retrieving a filing."""
        filing_id = sample_filing_data["accession_number"]
        
        # Test cached data
        mock_file_storage.load_cached_filing.return_value = {"processed_data": sample_filing_data}
        filing = filing_processor.get_filing(filing_id)
        
        assert filing is not None
        assert filing["processed_data"]["accession_number"] == filing_id
        
        # Test processed data
        mock_file_storage.load_cached_filing.return_value = None
        mock_file_storage.load_processed_filing.return_value = sample_filing_data
        filing = filing_processor.get_filing(filing_id)
        
        assert filing is not None
        assert filing["accession_number"] == filing_id

class TestFileStorage:
    """Test cases for file storage."""
    
    def test_save_raw_filing(self, mock_file_storage, temp_storage_dir, sample_filing_data):
        """Test saving raw filing data."""
        filing_id = sample_filing_data["accession_number"]
        
        mock_file_storage.save_raw_filing(
            filing_id=filing_id,
            content=sample_filing_data["text"],
            metadata=sample_filing_data
        )
        
        # Verify file was created
        raw_file = temp_storage_dir / "raw" / f"{filing_id}.json"
        assert raw_file.exists()
    
    def test_save_html_filing(self, mock_file_storage, temp_storage_dir, sample_filing_data):
        """Test saving HTML filing data."""
        filing_id = sample_filing_data["accession_number"]
        
        mock_file_storage.save_html_filing(
            filing_id=filing_id,
            html_content=sample_filing_data["html_content"],
            metadata=sample_filing_data
        )
        
        # Verify file was created
        html_file = temp_storage_dir / "html" / f"{filing_id}.html"
        assert html_file.exists()
    
    def test_save_processed_filing(self, mock_file_storage, temp_storage_dir, sample_filing_data, sample_chunks):
        """Test saving processed filing data."""
        filing_id = sample_filing_data["accession_number"]
        processed_data = {
            "filing_id": filing_id,
            "text": sample_filing_data["text"],
            "metadata": sample_filing_data,
            "chunks": sample_chunks.to_dict('records')
        }
        
        mock_file_storage.save_processed_filing(
            filing_id=filing_id,
            processed_data=processed_data,
            metadata=sample_filing_data
        )
        
        # Verify file was created
        processed_file = temp_storage_dir / "processed" / f"{filing_id}.json"
        assert processed_file.exists()
    
    def test_list_filings(self, mock_file_storage):
        """Test listing available filings."""
        # Test filtering by ticker
        filings = mock_file_storage.list_filings(ticker="AAPL")
        assert isinstance(filings, list)
        
        # Test filtering by year
        filings = mock_file_storage.list_filings(year="2023")
        assert isinstance(filings, list)
        
        # Test filtering by filing type
        filings = mock_file_storage.list_filings(filing_type="10-K")
        assert isinstance(filings, list) 