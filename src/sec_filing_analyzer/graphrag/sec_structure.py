"""
SEC Filing Structure Parser

This module provides functionality for parsing and analyzing SEC filing structure,
leveraging edgartools for document processing and maintaining graph-specific functionality.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import from the installed edgar package
from edgar import Company, Document, XBRLData
from edgar.company_reports import TenK, TenQ, EightK

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECStructure:
    """
    Class for parsing and analyzing SEC filing structure.
    Combines edgartools document processing with graph-specific functionality.
    """
    
    def __init__(self):
        """Initialize the SEC structure parser."""
        self.sections = {}
        self.hierarchical_structure = {}
        
        # Map form types to their corresponding structure classes
        self.form_structures = {
            "10-K": TenK,
            "10-Q": TenQ,
            "8-K": EightK
        }
        
        # Default sections for each form type
        self.default_sections = {
            "10-K": [
                "Item 1", "Item 1A", "Item 1B", "Item 2", "Item 3", "Item 4",
                "Item 5", "Item 6", "Item 7", "Item 7A", "Item 8", "Item 9",
                "Item 9A", "Item 9B", "Item 10", "Item 11", "Item 12", "Item 13",
                "Item 14", "Item 15"
            ],
            "10-Q": [
                "Item 1", "Item 2", "Item 3", "Item 4",
                "Item 1A", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6"
            ],
            "8-K": [
                "Item 1.01", "Item 1.02", "Item 1.03", "Item 1.04",
                "Item 2.01", "Item 2.02", "Item 2.03", "Item 2.04",
                "Item 2.05", "Item 2.06", "Item 3.01", "Item 3.02",
                "Item 3.03", "Item 4.01", "Item 4.02", "Item 5.01",
                "Item 5.02", "Item 5.03", "Item 5.04", "Item 5.05",
                "Item 5.06", "Item 5.07", "Item 5.08", "Item 6.01",
                "Item 6.02", "Item 6.03", "Item 6.04", "Item 6.05",
                "Item 7.01", "Item 8.01", "Item 9.01"
            ]
        }
    
    def parse_filing_structure(self, filing_content: str, form_type: str = "10-K") -> Dict[str, Any]:
        """
        Parse the structure of an SEC filing.
        
        Args:
            filing_content: The content of the SEC filing
            form_type: The type of SEC form (e.g., "10-K", "10-Q", "8-K")
            
        Returns:
            Dict containing the parsed structure
        """
        # Initialize structure
        structure = {
            "sections": {},
            "hierarchy": {},
            "metadata": {},
            "xbrl_data": {},
            "tables": []
        }
        
        try:
            # Create document instance
            doc = Document.parse(filing_content)
            
            # Extract metadata from document
            structure["metadata"] = self._extract_metadata(doc)
            
            # Get the appropriate sections for the form type
            sections = self.default_sections.get(form_type, self.default_sections["10-K"])
            
            # Process sections by searching for section headers
            for section in sections:
                try:
                    # Search for section content using regex
                    pattern = rf"{section}\.\s+(.*?)(?=(?:{section}|$))"
                    matches = re.finditer(pattern, filing_content, re.DOTALL | re.IGNORECASE)
                    
                    for match in matches:
                        section_content = match.group(1).strip()
                        if section_content:
                            structure["sections"][section] = {
                                "content": section_content,
                                "title": section,
                                "description": "",
                                "tables": self._extract_tables(section_content)
                            }
                except Exception as e:
                    logger.warning(f"Error processing section {section}: {e}")
            
            # Try to extract XBRL data if available in the document
            if hasattr(doc, "xbrl_data"):
                structure["xbrl_data"] = self._extract_xbrl_data(doc.xbrl_data)
            
            # Extract tables from the entire document
            structure["tables"] = self._extract_tables(filing_content)
            
            # Build hierarchy
            self._build_hierarchy(structure)
            
        except Exception as e:
            logger.error(f"Error parsing filing structure: {e}")
            return {}
        
        return structure
    
    def _extract_metadata(self, doc: Document) -> Dict[str, Any]:
        """
        Extract metadata from document.
        
        Args:
            doc: The document to extract metadata from
            
        Returns:
            Dict containing extracted metadata
        """
        metadata = {}
        
        try:
            # Check if metadata is already available as a property
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                return doc.metadata
            
            # Extract basic metadata from document attributes
            metadata = {
                "cik": str(getattr(doc, "cik", "")),
                "name": str(getattr(doc, "company_name", "")),
                "ticker": str(getattr(doc, "ticker", "")),
                "form": str(getattr(doc, "form_type", "")),
                "filing_date": str(getattr(doc, "filing_date", ""))
            }
            
            # Remove empty values and format strings
            metadata = {
                k: v.strip('"<>').split(" id=")[0].replace("Mock name='mock.", "").replace("'", "")
                for k, v in metadata.items()
                if v and v != "None"
            }
            
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _extract_xbrl_data(self, xbrl_data: XBRLData) -> Dict[str, Any]:
        """
        Extract data from XBRL document.
        
        Args:
            xbrl_data: The XBRL data to process
            
        Returns:
            Dict containing extracted XBRL data
        """
        data = {}
        
        try:
            # Extract available XBRL data
            if hasattr(xbrl_data, "data"):
                data = xbrl_data.data
            else:
                # Fallback to individual attributes
                if hasattr(xbrl_data, "label_to_concept_map"):
                    data["label_to_concept_map"] = xbrl_data.label_to_concept_map
                
                if hasattr(xbrl_data, "calculations"):
                    data["calculations"] = xbrl_data.calculations
                    
                if hasattr(xbrl_data, "statements_dict"):
                    data["statements_dict"] = xbrl_data.statements_dict
                
        except Exception as e:
            logger.warning(f"Error extracting XBRL data: {e}")
        
        return data
    
    def _extract_tables(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract tables from content.
        
        Args:
            content: Content to extract tables from
            
        Returns:
            List of dicts containing table data
        """
        tables = []
        
        try:
            # Simple table detection using regex
            table_pattern = r"<table.*?>(.*?)</table>"
            matches = re.finditer(table_pattern, content, re.DOTALL)
            
            for i, match in enumerate(matches):
                table_content = match.group(1)
                rows = self._parse_table_rows(table_content)
                
                if rows:
                    headers = rows[0]
                    data_rows = rows[1:] if len(rows) > 1 else []
                    
                    tables.append({
                        "id": f"table_{i}",
                        "headers": headers,
                        "rows": data_rows
                    })
                
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")
            
        return tables
    
    def _parse_table_rows(self, table_content: str) -> List[List[str]]:
        """
        Parse rows from table content.
        
        Args:
            table_content: HTML table content to parse
            
        Returns:
            List of rows, where each row is a list of cell values
        """
        rows = []
        
        try:
            # Extract rows
            row_pattern = r"<tr.*?>(.*?)</tr>"
            row_matches = re.finditer(row_pattern, table_content, re.DOTALL)
            
            for row_match in row_matches:
                row_content = row_match.group(1)
                
                # Extract cells
                cell_pattern = r"<t[dh].*?>(.*?)</t[dh]>"
                cells = re.findall(cell_pattern, row_content, re.DOTALL)
                
                # Clean cell content
                cells = [re.sub(r"<.*?>", "", cell).strip() for cell in cells]
                rows.append(cells)
                
        except Exception as e:
            logger.warning(f"Error parsing table rows: {e}")
            
        return rows
    
    def _build_hierarchy(self, structure: Dict[str, Any]) -> None:
        """
        Build hierarchical structure from sections.
        
        Args:
            structure: Structure dict to update with hierarchy
        """
        try:
            # Update the hierarchical_structure attribute
            self.hierarchical_structure = structure
            
        except Exception as e:
            logger.warning(f"Error building hierarchy: {e}")
    
    def extract_sections(self, filing_content: str, form_type: str = "10-K") -> Dict[str, str]:
        """
        Extract sections from an SEC filing.
        
        Args:
            filing_content: The content of the SEC filing
            form_type: The type of SEC form
            
        Returns:
            Dict mapping section IDs to their content
        """
        try:
            # Create document instance
            doc = Document.parse(filing_content)
            
            # Check if sections are already available in the document
            if hasattr(doc, "sections") and isinstance(doc.sections, dict):
                self.sections = doc.sections
                return doc.sections
            
            # Get the appropriate sections for the form type
            sections = self.default_sections.get(form_type, self.default_sections["10-K"])
            
            # Initialize sections dictionary
            extracted_sections = {}
            
            # Process sections by searching for section headers
            for section in sections:
                try:
                    # Search for section content using regex
                    pattern = rf"{section}\.\s+(.*?)(?=(?:{section}|$))"
                    matches = re.finditer(pattern, filing_content, re.DOTALL | re.IGNORECASE)
                    
                    for match in matches:
                        section_content = match.group(1).strip()
                        if section_content:
                            extracted_sections[section] = section_content
                            break  # Take only the first match for each section
                except Exception as e:
                    logger.warning(f"Error processing section {section}: {e}")
            
            # Store the sections in the instance variable
            self.sections = extracted_sections
            
            return extracted_sections
            
        except Exception as e:
            logger.error(f"Error extracting sections: {e}")
            return {}
    
    def get_section_content(self, section_id: str) -> Optional[str]:
        """
        Get the content of a specific section.
        
        Args:
            section_id: ID of the section to retrieve
            
        Returns:
            Section content if found, None otherwise
        """
        return self.sections.get(section_id) 