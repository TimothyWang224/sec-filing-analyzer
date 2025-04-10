"""
Neo4j Graph Enhancement Script

This script enhances the Neo4j graph structure to better support LLM agent queries by:
1. Adding section type classification to chunks
2. Creating temporal connections between filings
3. Implementing basic entity recognition
"""

import logging
import os
from typing import Dict, List, Tuple, Set
import re
from neo4j import GraphDatabase
from sec_filing_analyzer.config import neo4j_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Neo4jEnhancer:
    """Class to enhance Neo4j graph for LLM agent queries."""

    def __init__(
        self,
        url: str = None,
        username: str = None,
        password: str = None,
        database: str = None
    ):
        """Initialize the Neo4j enhancer."""
        # Get Neo4j credentials from parameters or environment variables
        self.uri = url or os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL") or neo4j_config.url
        self.user = username or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or neo4j_config.username
        self.pwd = password or os.getenv("NEO4J_PASSWORD") or neo4j_config.password
        self.db = database or os.getenv("NEO4J_DATABASE") or neo4j_config.database

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {str(e)}")
            raise

    def close(self):
        """Close the Neo4j connection."""
        if hasattr(self, 'driver'):
            self.driver.close()
            logger.info("Neo4j connection closed")

    def add_section_type_classification(self):
        """
        Add section type classification to chunks based on item numbers.
        
        Maps common SEC filing section identifiers to standardized section types.
        """
        # Define mapping from item numbers to section types
        item_to_section_type = {
            "Item 1": "Business",
            "Item 1A": "Risk Factors",
            "Item 1B": "Unresolved Staff Comments",
            "Item 2": "Properties",
            "Item 3": "Legal Proceedings",
            "Item 4": "Mine Safety Disclosures",
            "Item 5": "Market for Registrant's Common Equity",
            "Item 6": "Selected Financial Data",
            "Item 7": "Management's Discussion and Analysis",
            "Item 7A": "Quantitative and Qualitative Disclosures About Market Risk",
            "Item 8": "Financial Statements and Supplementary Data",
            "Item 9": "Changes in and Disagreements with Accountants",
            "Item 9A": "Controls and Procedures",
            "Item 9B": "Other Information",
            "Item 10": "Directors, Executive Officers and Corporate Governance",
            "Item 11": "Executive Compensation",
            "Item 12": "Security Ownership of Certain Beneficial Owners",
            "Item 13": "Certain Relationships and Related Transactions",
            "Item 14": "Principal Accounting Fees and Services",
            "Item 15": "Exhibits, Financial Statement Schedules"
        }
        
        with self.driver.session(database=self.db) as session:
            # First, add section_type property to chunks based on item property
            result = session.run("""
                MATCH (c:Chunk)
                WHERE c.item IS NOT NULL AND c.item <> ''
                RETURN DISTINCT c.item as item
            """)
            
            items = [record["item"] for record in result]
            
            for item in items:
                section_type = item_to_section_type.get(item, "Other")
                
                update_result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.item = $item
                    SET c.section_type = $section_type
                    RETURN count(c) as updated_count
                """, item=item, section_type=section_type)
                
                record = update_result.single()
                if record:
                    logger.info(f"Updated {record['updated_count']} chunks with section_type '{section_type}' for item '{item}'")
            
            # For chunks without an item property, try to infer from text content
            result = session.run("""
                MATCH (c:Chunk)
                WHERE (c.item IS NULL OR c.item = '') AND c.text CONTAINS 'Item '
                RETURN c.id as id, c.text as text
                LIMIT 1000
            """)
            
            pattern = re.compile(r'Item\s+(\d+[A-Z]?)\b')
            
            for record in result:
                chunk_id = record["id"]
                text = record["text"]
                
                # Try to find item number in text
                match = pattern.search(text)
                if match:
                    item_number = match.group(0)  # Full "Item X" match
                    section_type = item_to_section_type.get(item_number, "Other")
                    
                    update_result = session.run("""
                        MATCH (c:Chunk {id: $chunk_id})
                        SET c.inferred_item = $item_number,
                            c.section_type = $section_type
                        RETURN c.id
                    """, chunk_id=chunk_id, item_number=item_number, section_type=section_type)
                    
                    if update_result.single():
                        logger.info(f"Inferred item '{item_number}' and set section_type '{section_type}' for chunk {chunk_id}")
            
            # Count how many chunks have section_type property now
            count_result = session.run("""
                MATCH (c:Chunk)
                WHERE c.section_type IS NOT NULL
                RETURN count(c) as count
            """)
            
            record = count_result.single()
            if record:
                logger.info(f"Total chunks with section_type: {record['count']}")

    def create_temporal_connections(self):
        """
        Create temporal connections between filings from the same company.
        
        Adds NEXT_FILING and PREVIOUS_FILING relationships to establish chronology.
        """
        with self.driver.session(database=self.db) as session:
            # Get all companies
            result = session.run("""
                MATCH (c:Company)
                RETURN c.ticker as ticker
            """)
            
            companies = [record["ticker"] for record in result]
            
            for ticker in companies:
                # Get all filings for this company ordered by date
                result = session.run("""
                    MATCH (c:Company {ticker: $ticker})-[:FILED]->(f:Filing)
                    WHERE f.filing_date IS NOT NULL AND f.filing_date <> ''
                    RETURN f.id as id, f.filing_date as filing_date
                    ORDER BY f.filing_date
                """, ticker=ticker)
                
                filings = [(record["id"], record["filing_date"]) for record in result]
                
                if len(filings) < 2:
                    logger.info(f"Company {ticker} has fewer than 2 filings, skipping temporal connections")
                    continue
                
                # Create NEXT_FILING and PREVIOUS_FILING relationships
                for i in range(len(filings) - 1):
                    current_id, current_date = filings[i]
                    next_id, next_date = filings[i + 1]
                    
                    # Create relationship from earlier to later filing
                    result = session.run("""
                        MATCH (current:Filing {id: $current_id}), (next:Filing {id: $next_id})
                        MERGE (current)-[r:NEXT_FILING]->(next)
                        MERGE (next)-[r2:PREVIOUS_FILING]->(current)
                        RETURN current.id, next.id
                    """, current_id=current_id, next_id=next_id)
                    
                    if result.single():
                        logger.info(f"Created temporal connection between filings {current_id} ({current_date}) and {next_id} ({next_date})")
            
            # Count temporal relationships
            count_result = session.run("""
                MATCH ()-[r:NEXT_FILING]->()
                RETURN count(r) as count
            """)
            
            record = count_result.single()
            if record:
                logger.info(f"Total NEXT_FILING relationships: {record['count']}")
                
            count_result = session.run("""
                MATCH ()-[r:PREVIOUS_FILING]->()
                RETURN count(r) as count
            """)
            
            record = count_result.single()
            if record:
                logger.info(f"Total PREVIOUS_FILING relationships: {record['count']}")

    def implement_basic_entity_recognition(self):
        """
        Implement basic entity recognition for company mentions.
        
        Creates Entity nodes for companies and MENTIONS relationships to chunks.
        """
        # Define companies to look for
        companies = [
            {"name": "Apple", "ticker": "AAPL", "aliases": ["Apple Inc.", "Apple Computer"]},
            {"name": "Microsoft", "ticker": "MSFT", "aliases": ["Microsoft Corporation", "MSFT"]},
            {"name": "NVIDIA", "ticker": "NVDA", "aliases": ["NVIDIA Corporation", "NVDA"]},
            {"name": "Google", "ticker": "GOOGL", "aliases": ["Alphabet", "Alphabet Inc.", "Google LLC"]},
            {"name": "Amazon", "ticker": "AMZN", "aliases": ["Amazon.com", "Amazon.com Inc."]},
            {"name": "Meta", "ticker": "META", "aliases": ["Facebook", "Meta Platforms", "Facebook, Inc."]},
            {"name": "Tesla", "ticker": "TSLA", "aliases": ["Tesla, Inc.", "Tesla Motors"]},
            {"name": "Intel", "ticker": "INTC", "aliases": ["Intel Corporation"]},
            {"name": "AMD", "ticker": "AMD", "aliases": ["Advanced Micro Devices", "Advanced Micro Devices, Inc."]},
            {"name": "Qualcomm", "ticker": "QCOM", "aliases": ["Qualcomm Incorporated", "QUALCOMM"]}
        ]
        
        with self.driver.session(database=self.db) as session:
            # First, create Entity nodes for companies if they don't exist
            for company in companies:
                result = session.run("""
                    MERGE (e:Entity {ticker: $ticker, type: 'Company'})
                    SET e.name = $name
                    RETURN e.ticker
                """, ticker=company["ticker"], name=company["name"])
                
                if result.single():
                    logger.info(f"Created or updated Entity node for {company['name']} ({company['ticker']})")
            
            # For each company, find mentions in chunks
            for company in companies:
                # Create search patterns for this company
                search_terms = [company["name"], company["ticker"]] + company["aliases"]
                
                for term in search_terms:
                    # Find chunks that mention this term
                    result = session.run("""
                        MATCH (c:Chunk)
                        WHERE c.text CONTAINS $term
                        RETURN c.id as chunk_id
                        LIMIT 1000
                    """, term=term)
                    
                    chunk_ids = [record["chunk_id"] for record in result]
                    
                    if not chunk_ids:
                        continue
                    
                    # Create MENTIONS relationships
                    for chunk_id in chunk_ids:
                        result = session.run("""
                            MATCH (c:Chunk {id: $chunk_id}), (e:Entity {ticker: $ticker})
                            MERGE (c)-[r:MENTIONS]->(e)
                            ON CREATE SET r.term = $term
                            RETURN c.id, e.ticker
                        """, chunk_id=chunk_id, ticker=company["ticker"], term=term)
                        
                        if result.single():
                            logger.debug(f"Created MENTIONS relationship from chunk {chunk_id} to {company['ticker']} for term '{term}'")
                
                # Count mentions for this company
                count_result = session.run("""
                    MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity {ticker: $ticker})
                    RETURN count(r) as count
                """, ticker=company["ticker"])
                
                record = count_result.single()
                if record:
                    logger.info(f"Total mentions of {company['name']} ({company['ticker']}): {record['count']}")
            
            # Count total Entity nodes and MENTIONS relationships
            count_result = session.run("""
                MATCH (e:Entity)
                RETURN count(e) as count
            """)
            
            record = count_result.single()
            if record:
                logger.info(f"Total Entity nodes: {record['count']}")
                
            count_result = session.run("""
                MATCH ()-[r:MENTIONS]->()
                RETURN count(r) as count
            """)
            
            record = count_result.single()
            if record:
                logger.info(f"Total MENTIONS relationships: {record['count']}")

def main():
    """Main function to run the Neo4j graph enhancements."""
    enhancer = None
    try:
        enhancer = Neo4jEnhancer()
        
        # 1. Add section type classification
        logger.info("Adding section type classification...")
        enhancer.add_section_type_classification()
        
        # 2. Create temporal connections
        logger.info("Creating temporal connections...")
        enhancer.create_temporal_connections()
        
        # 3. Implement basic entity recognition
        logger.info("Implementing basic entity recognition...")
        enhancer.implement_basic_entity_recognition()
        
        logger.info("Graph enhancement completed successfully!")
        
    except Exception as e:
        logger.error(f"Error enhancing Neo4j graph: {str(e)}")
    finally:
        if enhancer:
            enhancer.close()

if __name__ == "__main__":
    main()
