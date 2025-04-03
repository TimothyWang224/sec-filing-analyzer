"""
SEC Entity Extractor

This module provides functionality for extracting and analyzing entities from SEC filings.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import re
from pathlib import Path
from collections import defaultdict

import spacy
from spacy.tokens import Doc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECEntities:
    """
    Class for extracting and analyzing entities from SEC filings.
    """
    
    def __init__(self):
        """Initialize the SEC entity extractor."""
        self.entities = {}
        self.relationships = {}
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Entity types to extract
        self.entity_types = {
            "ORG": "organization",
            "PERSON": "person",
            "GPE": "location",
            "DATE": "date",
            "MONEY": "financial",
            "PERCENT": "financial",
            "QUANTITY": "financial"
        }
        
        # Relationship patterns
        self.relationship_patterns = [
            (r"(\w+)\s+is\s+(?:a|an)\s+(\w+)", "is_a"),
            (r"(\w+)\s+has\s+(\w+)", "has"),
            (r"(\w+)\s+owns\s+(\w+)", "owns"),
            (r"(\w+)\s+operates\s+(\w+)", "operates"),
            (r"(\w+)\s+reports\s+to\s+(\w+)", "reports_to"),
            (r"(\w+)\s+manages\s+(\w+)", "manages"),
            (r"(\w+)\s+invests\s+in\s+(\w+)", "invests_in"),
            (r"(\w+)\s+partners\s+with\s+(\w+)", "partners_with")
        ]
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity = {
                    "id": f"{ent.label_}_{ent.text}",
                    "text": ent.text,
                    "type": self.entity_types[ent.label_],
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_
                }
                entities.append(entity)
                self.entities[entity["id"]] = entity
        
        return entities
    
    def identify_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify relationships between entities.
        
        Args:
            entities: List of entities to analyze
            
        Returns:
            List of identified relationships
        """
        relationships = []
        
        # Extract relationships using patterns
        for pattern, rel_type in self.relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity1, entity2 = match.groups()
                
                # Find matching entities
                e1 = self._find_matching_entity(entity1, entities)
                e2 = self._find_matching_entity(entity2, entities)
                
                if e1 and e2:
                    relationship = {
                        "from_node": e1["id"],
                        "to_node": e2["id"],
                        "type": rel_type,
                        "properties": {
                            "confidence": 1.0,
                            "source": "pattern_match"
                        }
                    }
                    relationships.append(relationship)
                    self.relationships[f"{e1['id']}_{rel_type}_{e2['id']}"] = relationship
        
        return relationships
    
    def get_entity_info(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific entity.
        
        Args:
            entity_id: ID of the entity to retrieve
            
        Returns:
            Entity information if found, None otherwise
        """
        return self.entities.get(entity_id)
    
    def _find_matching_entity(self, text: str, entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find an entity that matches the given text.
        
        Args:
            text: Text to match
            entities: List of entities to search through
            
        Returns:
            Matching entity if found, None otherwise
        """
        # Try exact match
        for entity in entities:
            if entity["text"].lower() == text.lower():
                return entity
        
        # Try partial match
        for entity in entities:
            if text.lower() in entity["text"].lower():
                return entity
        
        return None 