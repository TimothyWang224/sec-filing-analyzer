"""
Test suite for SEC filing ETL pipeline.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from edgar.files.htmltools import ChunkedDocument

from sec_filing_analyzer.data_retrieval.file_storage import FileStorage
from sec_filing_analyzer.data_retrieval.filing_processor import FilingProcessor
from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline
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
def mock_graph_store():
    """Create a mock graph store."""
    return Mock(spec=GraphStore)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    return Mock(spec=LlamaIndexVectorStore)


@pytest.fixture
def mock_filing_processor():
    """Create a mock filing processor."""
    return Mock(spec=FilingProcessor)


@pytest.fixture
def mock_file_storage(tmp_path):
    """Create a mock file storage."""
    return Mock(spec=FileStorage)


@pytest.fixture
def etl_pipeline(mock_graph_store, mock_vector_store, mock_filing_processor, mock_file_storage):
    """Create an ETL pipeline with mocked dependencies."""
    return SECFilingETLPipeline(
        graph_store=mock_graph_store,
        vector_store=mock_vector_store,
        filing_processor=mock_filing_processor,
        file_storage=mock_file_storage,
    )


class TestETLPipeline:
    """Test cases for ETL pipeline."""

    @patch("sec_filing_analyzer.pipeline.etl_pipeline.Company")
    def test_process_company(self, mock_company, etl_pipeline, sample_filing_data):
        """Test processing all filings for a company."""
        # Setup mock company
        mock_company_instance = Mock()
        mock_filing = Mock()
        mock_filing.accession_number = sample_filing_data["accession_number"]
        mock_filing.form = sample_filing_data["form"]
        mock_filing.filing_date = sample_filing_data["filing_date"]
        mock_filing.company = sample_filing_data["company"]
        mock_filing.ticker = sample_filing_data["ticker"]
        mock_filing.description = sample_filing_data["description"]
        mock_filing.url = sample_filing_data["url"]
        mock_filing.text = sample_filing_data["text"]
        mock_filing.download = Mock()
        mock_filing.download_html = Mock(return_value="<html>Sample HTML</html>")

        mock_company_instance.get_filings.return_value = [mock_filing]
        mock_company.return_value = mock_company_instance

        # Process company filings
        etl_pipeline.process_company(
            ticker="AAPL", filing_types=["10-K"], start_date="2023-01-01", end_date="2023-12-31"
        )

        # Verify company was created
        mock_company.assert_called_once_with("AAPL")

        # Verify filings were retrieved
        mock_company_instance.get_filings.assert_called_once_with(
            form_types=["10-K"], start_date="2023-01-01", end_date="2023-12-31"
        )

    def test_process_filing(self, etl_pipeline, sample_filing_data):
        """Test processing a single filing."""
        # Create mock filing
        mock_filing = Mock()
        mock_filing.accession_number = sample_filing_data["accession_number"]
        mock_filing.form = sample_filing_data["form"]
        mock_filing.filing_date = sample_filing_data["filing_date"]
        mock_filing.company = sample_filing_data["company"]
        mock_filing.ticker = sample_filing_data["ticker"]
        mock_filing.description = sample_filing_data["description"]
        mock_filing.url = sample_filing_data["url"]
        mock_filing.text = sample_filing_data["text"]
        mock_filing.download = Mock()
        mock_filing.download_html = Mock(return_value="<html>Sample HTML</html>")

        # Setup mock filing processor
        etl_pipeline.filing_processor.process_filing.return_value = {
            "filing_id": mock_filing.accession_number,
            "text": mock_filing.text,
            "metadata": sample_filing_data,
        }

        # Process filing
        etl_pipeline.process_filing(mock_filing)

        # Verify filing was processed
        etl_pipeline.filing_processor.process_filing.assert_called_once()

        # Verify file storage was used
        etl_pipeline.file_storage.save_raw_filing.assert_called_once()
        etl_pipeline.file_storage.save_processed_filing.assert_called_once()

    @patch("sec_filing_analyzer.pipeline.etl_pipeline.OpenAIEmbedding")
    def test_generate_embeddings(self, mock_openai_embedding, etl_pipeline, sample_filing_data):
        """Test generating embeddings for filing content."""
        # Create mock filing
        mock_filing = Mock()
        mock_filing.accession_number = sample_filing_data["accession_number"]
        mock_filing.form = sample_filing_data["form"]
        mock_filing.filing_date = sample_filing_data["filing_date"]
        mock_filing.company = sample_filing_data["company"]
        mock_filing.ticker = sample_filing_data["ticker"]
        mock_filing.description = sample_filing_data["description"]
        mock_filing.url = sample_filing_data["url"]
        mock_filing.text = sample_filing_data["text"]
        mock_filing.download = Mock()
        mock_filing.download_html = Mock(return_value="<html>Sample HTML</html>")

        # Setup mock embedding model
        mock_embedding_instance = Mock()
        mock_embedding_instance.get_text_embedding.return_value = [0.1, 0.2, 0.3]
        mock_openai_embedding.return_value = mock_embedding_instance

        # Replace the embedding model in the pipeline
        etl_pipeline.embedding_model = mock_embedding_instance

        # Process filing with embeddings
        etl_pipeline.process_filing(mock_filing)

        # Verify embeddings were generated
        assert mock_embedding_instance.get_text_embedding.call_count >= 1

    def test_error_handling(self, etl_pipeline, sample_filing_data):
        """Test error handling during processing."""
        # Create mock filing
        mock_filing = Mock()
        mock_filing.accession_number = sample_filing_data["accession_number"]
        mock_filing.form = sample_filing_data["form"]
        mock_filing.filing_date = sample_filing_data["filing_date"]
        mock_filing.company = sample_filing_data["company"]
        mock_filing.ticker = sample_filing_data["ticker"]
        mock_filing.description = sample_filing_data["description"]
        mock_filing.url = sample_filing_data["url"]
        mock_filing.text = sample_filing_data["text"]
        mock_filing.download = Mock()
        mock_filing.download_html = Mock(side_effect=Exception("HTML download error"))

        # Process filing (should not raise exception)
        etl_pipeline.process_filing(mock_filing)

        # Verify error was logged but processing continued
        assert mock_filing.download_html.called

    @pytest.mark.parametrize("filing_type", ["10-K", "10-Q", "8-K"])
    def test_different_filing_types(self, etl_pipeline, filing_type, sample_filing_data):
        """Test processing different filing types."""
        # Create mock filing
        mock_filing = Mock()
        mock_filing.accession_number = sample_filing_data["accession_number"]
        mock_filing.form = filing_type
        mock_filing.filing_date = sample_filing_data["filing_date"]
        mock_filing.company = sample_filing_data["company"]
        mock_filing.ticker = sample_filing_data["ticker"]
        mock_filing.description = sample_filing_data["description"]
        mock_filing.url = sample_filing_data["url"]
        mock_filing.text = sample_filing_data["text"]
        mock_filing.download = Mock()
        mock_filing.download_html = Mock(return_value="<html>Sample HTML</html>")

        # Process filing
        etl_pipeline.process_filing(mock_filing)

        # Verify filing was processed correctly
        etl_pipeline.filing_processor.process_filing.assert_called_once()
        processed_data = etl_pipeline.filing_processor.process_filing.call_args[0][0]
        assert processed_data["metadata"]["form"] == filing_type

    def test_chunk_processing(self, etl_pipeline, sample_filing_data, sample_chunks):
        """Test processing filing with chunks."""
        # Create mock filing
        mock_filing = Mock()
        mock_filing.accession_number = sample_filing_data["accession_number"]
        mock_filing.form = sample_filing_data["form"]
        mock_filing.filing_date = sample_filing_data["filing_date"]
        mock_filing.company = sample_filing_data["company"]
        mock_filing.ticker = sample_filing_data["ticker"]
        mock_filing.description = sample_filing_data["description"]
        mock_filing.url = sample_filing_data["url"]
        mock_filing.text = sample_filing_data["text"]
        mock_filing.download = Mock()

        # Create mock HTML content with chunks
        html_content = "<html><body>"
        for _, chunk in sample_chunks.iterrows():
            html_content += f"<div>{chunk['text']}</div>"
        html_content += "</body></html>"
        mock_filing.download_html = Mock(return_value=html_content)

        # Mock ChunkedDocument
        mock_chunked_doc = Mock()
        mock_chunked_doc.as_dataframe.return_value = sample_chunks
        mock_chunked_doc.list_items.return_value = []
        mock_chunked_doc.average_chunk_size.return_value = 100

        with patch("sec_filing_analyzer.pipeline.etl_pipeline.ChunkedDocument", return_value=mock_chunked_doc):
            # Process filing with chunks
            etl_pipeline.process_filing(mock_filing)

            # Verify chunks were processed
            etl_pipeline.filing_processor.process_filing.assert_called_once()
            processed_data = etl_pipeline.filing_processor.process_filing.call_args[0][0]
            assert "chunks" in processed_data
            assert len(processed_data["chunks"]) == len(sample_chunks)
