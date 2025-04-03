"""
SEC Filing Structure Parser

This module provides functionality for parsing and analyzing SEC filing structure.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECStructure:
    """
    Class for parsing and analyzing SEC filing structure.
    """
    
    def __init__(self):
        """Initialize the SEC structure parser."""
        self.sections = {}
        self.hierarchical_structure = {}
        
        # Common SEC section patterns
        self.section_patterns = {
            "item_1": r"Item\s+1\.?\s*[-–]?\s*Business",
            "item_1a": r"Item\s+1A\.?\s*[-–]?\s*Risk\s+Factors",
            "item_2": r"Item\s+2\.?\s*[-–]?\s*Properties",
            "item_3": r"Item\s+3\.?\s*[-–]?\s*Legal\s+Proceedings",
            "item_4": r"Item\s+4\.?\s*[-–]?\s*Mine\s+Safety\s+Disclosures",
            "item_5": r"Item\s+5\.?\s*[-–]?\s*Market\s+for\s+Registrant's\s+Common\s+Equity",
            "item_6": r"Item\s+6\.?\s*[-–]?\s*Selected\s+Financial\s+Data",
            "item_7": r"Item\s+7\.?\s*[-–]?\s*Management's\s+Discussion\s+and\s+Analysis",
            "item_7a": r"Item\s+7A\.?\s*[-–]?\s*Quantitative\s+and\s+Qualitative\s+Disclosures",
            "item_8": r"Item\s+8\.?\s*[-–]?\s*Financial\s+Statements\s+and\s+Supplementary\s+Data",
            "item_9": r"Item\s+9\.?\s*[-–]?\s*Changes\s+in\s+and\s+Disagreements\s+with\s+Accountants",
            "item_9a": r"Item\s+9A\.?\s*[-–]?\s*Controls\s+and\s+Procedures",
            "item_9b": r"Item\s+9B\.?\s*[-–]?\s*Other\s+Information",
            "item_10": r"Item\s+10\.?\s*[-–]?\s*Directors,\s+Executive\s+Officers",
            "item_11": r"Item\s+11\.?\s*[-–]?\s*Executive\s+Compensation",
            "item_12": r"Item\s+12\.?\s*[-–]?\s*Security\s+Ownership\s+of\s+Certain\s+Beneficial\s+Owners",
            "item_13": r"Item\s+13\.?\s*[-–]?\s*Certain\s+Relationships\s+and\s+Related\s+Transactions",
            "item_14": r"Item\s+14\.?\s*[-–]?\s*Principal\s+Accountant\s+Fees\s+and\s+Services",
            "item_15": r"Item\s+15\.?\s*[-–]?\s*Exhibits\s+and\s+Financial\s+Statement\s+Schedules",
            "item_16": r"Item\s+16\.?\s*[-–]?\s*Form\s+10-K\s+Summary"
        }
    
    def parse_filing_structure(self, filing_content: str) -> Dict[str, Any]:
        """
        Parse the structure of an SEC filing.
        
        Args:
            filing_content: The content of the SEC filing
            
        Returns:
            Dict containing the parsed structure
        """
        # Initialize structure
        structure = {
            "sections": {},
            "hierarchy": {},
            "metadata": {}
        }
        
        # Extract sections
        for section_id, pattern in self.section_patterns.items():
            match = re.search(pattern, filing_content, re.IGNORECASE)
            if match:
                start_pos = match.start()
                # Find the next section start
                next_pos = len(filing_content)
                for other_pattern in self.section_patterns.values():
                    if other_pattern != pattern:
                        next_match = re.search(other_pattern, filing_content[start_pos+1:], re.IGNORECASE)
                        if next_match:
                            next_pos = min(next_pos, start_pos + 1 + next_match.start())
                
                # Extract section content
                section_content = filing_content[start_pos:next_pos].strip()
                structure["sections"][section_id] = {
                    "start": start_pos,
                    "end": next_pos,
                    "content": section_content
                }
        
        # Build hierarchy
        self._build_hierarchy(structure)
        
        return structure
    
    def extract_sections(self, filing_content: str) -> Dict[str, str]:
        """
        Extract sections from an SEC filing.
        
        Args:
            filing_content: The content of the SEC filing
            
        Returns:
            Dict mapping section IDs to their content
        """
        sections = {}
        
        # Extract sections using patterns
        for section_id, pattern in self.section_patterns.items():
            match = re.search(pattern, filing_content, re.IGNORECASE)
            if match:
                start_pos = match.start()
                # Find the next section start
                next_pos = len(filing_content)
                for other_pattern in self.section_patterns.values():
                    if other_pattern != pattern:
                        next_match = re.search(other_pattern, filing_content[start_pos+1:], re.IGNORECASE)
                        if next_match:
                            next_pos = min(next_pos, start_pos + 1 + next_match.start())
                
                # Extract section content
                section_content = filing_content[start_pos:next_pos].strip()
                sections[section_id] = section_content
        
        return sections
    
    def get_section_content(self, section_id: str) -> Optional[str]:
        """
        Get the content of a specific section.
        
        Args:
            section_id: ID of the section to retrieve
            
        Returns:
            Section content if found, None otherwise
        """
        return self.sections.get(section_id)
    
    def _build_hierarchy(self, structure: Dict[str, Any]) -> None:
        """
        Build the hierarchical structure of the filing.
        
        Args:
            structure: The filing structure to build hierarchy for
        """
        # Sort sections by start position
        sorted_sections = sorted(
            structure["sections"].items(),
            key=lambda x: x[1]["start"]
        )
        
        # Build hierarchy
        hierarchy = {}
        current_level = hierarchy
        
        for section_id, section_info in sorted_sections:
            # Check if this is a subsection
            if "." in section_id:
                parent_id = section_id.rsplit(".", 1)[0]
                if parent_id in current_level:
                    current_level[parent_id]["subsections"][section_id] = section_info
                else:
                    current_level[section_id] = {
                        "info": section_info,
                        "subsections": {}
                    }
            else:
                current_level[section_id] = {
                    "info": section_info,
                    "subsections": {}
                }
        
        structure["hierarchy"] = hierarchy 