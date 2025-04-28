"""
Tests for the SECStructure class in the graphrag module.
"""

from unittest.mock import Mock, patch

import pytest

from sec_filing_analyzer.graphrag.sec_structure import SECStructure

# Sample test data
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

SAMPLE_METADATA = {
    "cik": "0000320193",
    "name": "Apple Inc.",
    "ticker": "AAPL",
    "form": "10-K",
    "filing_date": "2023-10-27",
}

SAMPLE_XBRL_DATA = {
    "revenue": 394328000000,
    "net_income": 96995000000,
    "total_assets": 352755000000,
    "total_liabilities": 290437000000,
}

SAMPLE_TABLE_DATA = [
    {
        "headers": ["Item", "2023", "2022", "2021"],
        "rows": [
            ["Revenue", "$394,328", "$365,817", "$365,697"],
            ["Net Income", "$96,995", "$99,803", "$94,680"],
        ],
    }
]

SAMPLE_SECTIONS = {
    "Item 1": "Business Overview",
    "Item 1A": "Risk Factors",
    "Item 2": "Properties",
    "Item 3": "Legal Proceedings",
    "Item 4": "Mine Safety Disclosures",
    "Item 5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "Item 6": "Selected Financial Data",
    "Item 7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "Item 7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "Item 8": "Financial Statements and Supplementary Data",
    "Item 9": "Changes in and Disagreements With Accountants on Accounting and Financial Disclosure",
    "Item 9A": "Controls and Procedures",
    "Item 9B": "Other Information",
    "Item 10": "Directors, Executive Officers and Corporate Governance",
    "Item 11": "Executive Compensation",
    "Item 12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "Item 13": "Certain Relationships and Related Transactions, and Director Independence",
    "Item 14": "Principal Accountant Fees and Services",
    "Item 15": "Exhibits, Financial Statement Schedules",
}


@pytest.fixture
def sec_structure():
    """Create a SECStructure instance."""
    return SECStructure()


@pytest.fixture
def mock_document():
    """Create a mock Document object."""
    mock = Mock()
    mock.metadata = SAMPLE_METADATA
    return mock


@pytest.fixture
def mock_xbrl_data():
    """Create a mock XBRLData object."""
    mock = Mock()
    mock.data = SAMPLE_XBRL_DATA
    return mock


@pytest.fixture
def mock_tenk():
    """Create a mock TenK object."""
    mock = Mock()
    mock.sections = SAMPLE_SECTIONS
    return mock


class TestSECStructure:
    """Test cases for the SECStructure class."""

    def test_init(self, sec_structure):
        """Test initialization of SECStructure."""
        assert isinstance(sec_structure, SECStructure)
        assert sec_structure.sections == {}
        assert sec_structure.hierarchical_structure == {}
        assert "10-K" in sec_structure.form_structures
        assert "10-Q" in sec_structure.form_structures
        assert "8-K" in sec_structure.form_structures
        assert "10-K" in sec_structure.default_sections
        assert "10-Q" in sec_structure.default_sections

    @patch("sec_filing_analyzer.graphrag.sec_structure.Document")
    def test_parse_filing_structure_success(
        self, mock_document_class, sec_structure, mock_document, mock_xbrl_data
    ):
        """Test successful parsing of filing structure."""
        # Setup mocks
        mock_document_class.parse.return_value = mock_document
        with (
            patch.object(
                sec_structure, "_extract_metadata", return_value=SAMPLE_METADATA
            ),
            patch.object(
                sec_structure, "_extract_xbrl_data", return_value=SAMPLE_XBRL_DATA
            ),
            patch.object(
                sec_structure, "_extract_tables", return_value=SAMPLE_TABLE_DATA
            ),
            patch.object(sec_structure, "_build_hierarchy"),
        ):
            # Call the method
            result = sec_structure.parse_filing_structure(SAMPLE_FILING_CONTENT)

            # Verify the result
            assert result["metadata"] == SAMPLE_METADATA
            assert result["xbrl_data"] == SAMPLE_XBRL_DATA
            assert result["tables"] == SAMPLE_TABLE_DATA

            # Verify the mocks were called correctly
            mock_document_class.parse.assert_called_once_with(SAMPLE_FILING_CONTENT)

    @patch("sec_filing_analyzer.graphrag.sec_structure.Document")
    def test_parse_filing_structure_exception(self, mock_document_class, sec_structure):
        """Test handling of exceptions when parsing filing structure."""
        # Setup mock to raise an exception
        mock_document_class.parse.side_effect = Exception("Test exception")

        # Call the method
        result = sec_structure.parse_filing_structure(SAMPLE_FILING_CONTENT)

        # Verify the result
        assert result == {}

        # Verify the mock was called correctly
        mock_document_class.parse.assert_called_once_with(SAMPLE_FILING_CONTENT)

    def test_extract_metadata(self, sec_structure, mock_document):
        """Test extraction of metadata from document."""
        # Call the method
        result = sec_structure._extract_metadata(mock_document)

        # Verify the result
        assert result == SAMPLE_METADATA

    def test_extract_xbrl_data(self, sec_structure, mock_xbrl_data):
        """Test extraction of XBRL data."""
        # Call the method
        result = sec_structure._extract_xbrl_data(mock_xbrl_data)

        # Verify the result
        assert result == SAMPLE_XBRL_DATA

    def test_extract_tables(self, sec_structure):
        """Test extraction of tables from content."""
        # Sample table content
        table_content = """
        <table>
        <tr><th>Item</th><th>2023</th><th>2022</th><th>2021</th></tr>
        <tr><td>Revenue</td><td>$394,328</td><td>$365,817</td><td>$365,697</td></tr>
        <tr><td>Net Income</td><td>$96,995</td><td>$99,803</td><td>$94,680</td></tr>
        </table>
        """

        # Call the method
        result = sec_structure._extract_tables(table_content)

        # Verify the result
        assert len(result) > 0
        assert "headers" in result[0]
        assert "rows" in result[0]

    def test_parse_table_rows(self, sec_structure):
        """Test parsing of table rows."""
        # Sample table content
        table_content = """
        <tr><th>Item</th><th>2023</th><th>2022</th><th>2021</th></tr>
        <tr><td>Revenue</td><td>$394,328</td><td>$365,817</td><td>$365,697</td></tr>
        <tr><td>Net Income</td><td>$96,995</td><td>$99,803</td><td>$94,680</td></tr>
        """

        # Call the method
        result = sec_structure._parse_table_rows(table_content)

        # Verify the result
        assert len(result) > 0
        assert len(result[0]) == 4  # 4 columns

    def test_build_hierarchy(self, sec_structure):
        """Test building of hierarchical structure."""
        # Sample structure
        structure = {
            "metadata": SAMPLE_METADATA,
            "xbrl_data": SAMPLE_XBRL_DATA,
            "tables": SAMPLE_TABLE_DATA,
        }

        # Call the method
        sec_structure._build_hierarchy(structure)

        # Verify the result
        assert sec_structure.hierarchical_structure == structure

    @patch("sec_filing_analyzer.graphrag.sec_structure.Document")
    def test_extract_sections_success(
        self, mock_document_class, sec_structure, mock_tenk
    ):
        """Test successful extraction of sections."""
        # Setup mocks
        mock_document_class.parse.return_value = mock_tenk

        # Call the method
        result = sec_structure.extract_sections(SAMPLE_FILING_CONTENT)

        # Verify the result
        assert result == SAMPLE_SECTIONS

        # Verify the mock was called correctly
        mock_document_class.parse.assert_called_once_with(SAMPLE_FILING_CONTENT)

    @patch("sec_filing_analyzer.graphrag.sec_structure.Document")
    def test_extract_sections_exception(self, mock_document_class, sec_structure):
        """Test handling of exceptions when extracting sections."""
        # Setup mock to raise an exception
        mock_document_class.parse.side_effect = Exception("Test exception")

        # Call the method
        result = sec_structure.extract_sections(SAMPLE_FILING_CONTENT)

        # Verify the result
        assert result == {}

        # Verify the mock was called correctly
        mock_document_class.parse.assert_called_once_with(SAMPLE_FILING_CONTENT)

    def test_get_section_content(self, sec_structure):
        """Test getting section content."""
        # Setup test data
        sec_structure.sections = SAMPLE_SECTIONS

        # Call the method
        result = sec_structure.get_section_content("Item 1")

        # Verify the result
        assert result == "Business Overview"

    def test_get_section_content_not_found(self, sec_structure):
        """Test getting section content when section is not found."""
        # Setup test data
        sec_structure.sections = SAMPLE_SECTIONS

        # Call the method
        result = sec_structure.get_section_content("Non-existent Section")

        # Verify the result
        assert result is None
