"""
Tests for the SECEntities class in the graphrag module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from sec_filing_analyzer.graphrag.sec_entities import SECEntities

# Sample test data
SAMPLE_COMPANY_DATA = {
    "cik": "0000320193",
    "name": "Apple Inc.",
    "tickers": ["AAPL"],
    "sic": "3571",
    "industry": "Computer Hardware",
    "address": "One Apple Park Way, Cupertino, CA 95014",
    "phone": "408-996-1010",
    "website": "https://www.apple.com",
    "fiscal_year_end": "09-30"
}

SAMPLE_ENTITY_DATA = {
    "cik": "0000320193",
    "name": "Apple Inc.",
    "filings": [
        {"accession_number": "0000320193-23-000077", "form": "10-K", "filing_date": "2023-10-27"},
        {"accession_number": "0000320193-23-000078", "form": "10-Q", "filing_date": "2023-11-15"}
    ]
}

SAMPLE_FILING_CONTENT = """
<SEC-DOCUMENT>
<SEC-HEADER>
<COMPANY-DATA>
<CIK>0000320193</CIK>
<CONFORMED-NAME>Apple Inc.</CONFORMED-NAME>
<TICKER>AAPL</TICKER>
</COMPANY-DATA>
</SEC-HEADER>
<DOCUMENT>
<TYPE>10-K</TYPE>
<SEQUENCE>1</SEQUENCE>
<FILENAME>aapl-20230930.htm</FILENAME>
<DESCRIPTION>Annual report</DESCRIPTION>
<TEXT>
This is a sample 10-K filing for Apple Inc.
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""

@pytest.fixture
def sec_entities():
    """Create a SECEntities instance."""
    return SECEntities()

@pytest.fixture
def mock_company():
    """Create a mock Company object."""
    mock = Mock()
    mock.cik = SAMPLE_COMPANY_DATA["cik"]
    mock.name = SAMPLE_COMPANY_DATA["name"]
    mock.tickers = SAMPLE_COMPANY_DATA["tickers"]
    mock.sic = SAMPLE_COMPANY_DATA["sic"]
    mock.industry = SAMPLE_COMPANY_DATA["industry"]
    mock.address = SAMPLE_COMPANY_DATA["address"]
    mock.phone = SAMPLE_COMPANY_DATA["phone"]
    mock.website = SAMPLE_COMPANY_DATA["website"]
    mock.fiscal_year_end = SAMPLE_COMPANY_DATA["fiscal_year_end"]
    return mock

@pytest.fixture
def mock_entity_data():
    """Create a mock EntityData object."""
    mock = Mock()
    mock.cik = SAMPLE_ENTITY_DATA["cik"]
    mock.name = SAMPLE_ENTITY_DATA["name"]
    mock.filings = SAMPLE_ENTITY_DATA["filings"]
    return mock

@pytest.fixture
def mock_document():
    """Create a mock Document object."""
    mock = Mock()
    mock.metadata = {
        "cik": SAMPLE_COMPANY_DATA["cik"],
        "name": SAMPLE_COMPANY_DATA["name"],
        "ticker": SAMPLE_COMPANY_DATA["tickers"][0],
        "form": "10-K",
        "filing_date": "2023-10-27"
    }
    return mock

class TestSECEntities:
    """Test cases for the SECEntities class."""
    
    def test_init(self, sec_entities):
        """Test initialization of SECEntities."""
        assert isinstance(sec_entities, SECEntities)
    
    @patch('sec_filing_analyzer.graphrag.sec_entities.find_company')
    def test_get_company_data_success(self, mock_find_company, sec_entities, mock_company, mock_entity_data):
        """Test successful retrieval of company data."""
        # Setup mocks
        mock_find_company.return_value = mock_company
        with patch.object(sec_entities, '_get_entity_data', return_value=mock_entity_data):
            # Call the method
            result = sec_entities.get_company_data("AAPL")
            
            # Verify the result
            assert result is not None
            assert result["cik"] == SAMPLE_COMPANY_DATA["cik"]
            assert result["name"] == SAMPLE_COMPANY_DATA["name"]
            assert result["tickers"] == SAMPLE_COMPANY_DATA["tickers"]
            assert result["entity_data"] == mock_entity_data
            
            # Verify the mock was called correctly
            mock_find_company.assert_called_once_with("AAPL")
    
    @patch('sec_filing_analyzer.graphrag.sec_entities.find_company')
    def test_get_company_data_not_found(self, mock_find_company, sec_entities):
        """Test when company data is not found."""
        # Setup mock to return None
        mock_find_company.return_value = None
        
        # Call the method
        result = sec_entities.get_company_data("INVALID")
        
        # Verify the result
        assert result is None
        
        # Verify the mock was called correctly
        mock_find_company.assert_called_once_with("INVALID")
    
    @patch('sec_filing_analyzer.graphrag.sec_entities.find_company')
    def test_get_company_data_exception(self, mock_find_company, sec_entities):
        """Test handling of exceptions when getting company data."""
        # Setup mock to raise an exception
        mock_find_company.side_effect = Exception("Test exception")
        
        # Call the method
        result = sec_entities.get_company_data("AAPL")
        
        # Verify the result
        assert result is None
        
        # Verify the mock was called correctly
        mock_find_company.assert_called_once_with("AAPL")
    
    def test_format_company_data(self, sec_entities, mock_company, mock_entity_data):
        """Test formatting of company data."""
        # Setup mock for _get_entity_data
        with patch.object(sec_entities, '_get_entity_data', return_value=mock_entity_data):
            # Call the method
            result = sec_entities._format_company_data(mock_company)
            
            # Verify the result
            assert result["cik"] == SAMPLE_COMPANY_DATA["cik"]
            assert result["name"] == SAMPLE_COMPANY_DATA["name"]
            assert result["tickers"] == SAMPLE_COMPANY_DATA["tickers"]
            assert result["sic"] == SAMPLE_COMPANY_DATA["sic"]
            assert result["industry"] == SAMPLE_COMPANY_DATA["industry"]
            assert result["address"] == SAMPLE_COMPANY_DATA["address"]
            assert result["phone"] == SAMPLE_COMPANY_DATA["phone"]
            assert result["website"] == SAMPLE_COMPANY_DATA["website"]
            assert result["fiscal_year_end"] == SAMPLE_COMPANY_DATA["fiscal_year_end"]
            assert result["entity_data"] == mock_entity_data
    
    @patch('sec_filing_analyzer.graphrag.sec_entities.get_entity_submissions')
    def test_get_entity_data_success(self, mock_get_entity_submissions, sec_entities, mock_entity_data):
        """Test successful retrieval of entity data."""
        # Setup mock
        mock_get_entity_submissions.return_value = mock_entity_data
        
        # Call the method
        result = sec_entities._get_entity_data(SAMPLE_COMPANY_DATA["cik"])
        
        # Verify the result
        assert result == mock_entity_data
        
        # Verify the mock was called correctly
        mock_get_entity_submissions.assert_called_once_with(SAMPLE_COMPANY_DATA["cik"])
    
    @patch('sec_filing_analyzer.graphrag.sec_entities.get_entity_submissions')
    def test_get_entity_data_exception(self, mock_get_entity_submissions, sec_entities):
        """Test handling of exceptions when getting entity data."""
        # Setup mock to raise an exception
        mock_get_entity_submissions.side_effect = Exception("Test exception")
        
        # Call the method
        result = sec_entities._get_entity_data(SAMPLE_COMPANY_DATA["cik"])
        
        # Verify the result
        assert result is None
        
        # Verify the mock was called correctly
        mock_get_entity_submissions.assert_called_once_with(SAMPLE_COMPANY_DATA["cik"])
    
    @patch('sec_filing_analyzer.graphrag.sec_entities.Document')
    def test_get_filing_metadata_success(self, mock_document_class, sec_entities, mock_document):
        """Test successful extraction of filing metadata."""
        # Setup mock
        mock_document_class.parse.return_value = mock_document
        
        # Call the method
        result = sec_entities.get_filing_metadata(SAMPLE_FILING_CONTENT)
        
        # Verify the result
        assert result == mock_document.metadata
        
        # Verify the mock was called correctly
        mock_document_class.parse.assert_called_once_with(SAMPLE_FILING_CONTENT)
    
    @patch('sec_filing_analyzer.graphrag.sec_entities.Document')
    def test_get_filing_metadata_no_metadata(self, mock_document_class, sec_entities):
        """Test when document has no metadata."""
        # Setup mock
        mock_doc = Mock()
        mock_doc.metadata = None
        mock_document_class.parse.return_value = mock_doc
        
        # Call the method
        result = sec_entities.get_filing_metadata(SAMPLE_FILING_CONTENT)
        
        # Verify the result
        assert result == {}
        
        # Verify the mock was called correctly
        mock_document_class.parse.assert_called_once_with(SAMPLE_FILING_CONTENT)
    
    @patch('sec_filing_analyzer.graphrag.sec_entities.Document')
    def test_get_filing_metadata_exception(self, mock_document_class, sec_entities):
        """Test handling of exceptions when getting filing metadata."""
        # Setup mock to raise an exception
        mock_document_class.parse.side_effect = Exception("Test exception")
        
        # Call the method
        result = sec_entities.get_filing_metadata(SAMPLE_FILING_CONTENT)
        
        # Verify the result
        assert result == {}
        
        # Verify the mock was called correctly
        mock_document_class.parse.assert_called_once_with(SAMPLE_FILING_CONTENT)
    
    @patch.object(SECEntities, 'get_filing_metadata')
    @patch.object(SECEntities, 'get_company_data')
    def test_extract_entities_with_metadata(self, mock_get_company_data, mock_get_filing_metadata, sec_entities, mock_company):
        """Test extraction of entities from filing with metadata."""
        # Setup mocks
        mock_get_filing_metadata.return_value = {"cik": SAMPLE_COMPANY_DATA["cik"]}
        mock_get_company_data.return_value = SAMPLE_COMPANY_DATA
        
        # Call the method
        result = sec_entities.extract_entities(SAMPLE_FILING_CONTENT)
        
        # Verify the result
        assert len(result) == 1
        assert result[0] == SAMPLE_COMPANY_DATA
        
        # Verify the mocks were called correctly
        mock_get_filing_metadata.assert_called_once_with(SAMPLE_FILING_CONTENT)
        mock_get_company_data.assert_called_once_with(SAMPLE_COMPANY_DATA["cik"])
    
    @patch.object(SECEntities, 'get_filing_metadata')
    @patch.object(SECEntities, 'get_company_data')
    def test_extract_entities_no_metadata(self, mock_get_company_data, mock_get_filing_metadata, sec_entities):
        """Test extraction of entities from filing without metadata."""
        # Setup mocks
        mock_get_filing_metadata.return_value = {}
        
        # Call the method
        result = sec_entities.extract_entities(SAMPLE_FILING_CONTENT)
        
        # Verify the result
        assert len(result) == 0
        
        # Verify the mocks were called correctly
        mock_get_filing_metadata.assert_called_once_with(SAMPLE_FILING_CONTENT)
        mock_get_company_data.assert_not_called()
    
    @patch.object(SECEntities, 'get_filing_metadata')
    @patch.object(SECEntities, 'get_company_data')
    def test_extract_entities_company_not_found(self, mock_get_company_data, mock_get_filing_metadata, sec_entities):
        """Test extraction of entities when company is not found."""
        # Setup mocks
        mock_get_filing_metadata.return_value = {"cik": SAMPLE_COMPANY_DATA["cik"]}
        mock_get_company_data.return_value = None
        
        # Call the method
        result = sec_entities.extract_entities(SAMPLE_FILING_CONTENT)
        
        # Verify the result
        assert len(result) == 0
        
        # Verify the mocks were called correctly
        mock_get_filing_metadata.assert_called_once_with(SAMPLE_FILING_CONTENT)
        mock_get_company_data.assert_called_once_with(SAMPLE_COMPANY_DATA["cik"]) 