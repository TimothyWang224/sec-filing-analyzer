"""
Filing Processor

Handles processing of SEC filings into chunks and embeddings.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from rich.console import Console

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import SimpleVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class FilingProcessor:
    """Processes SEC filings into chunks and embeddings."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "text-embedding-3-small"
    ):
        """Initialize the processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: OpenAI embedding model to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Initialize components
        self.llm = OpenAI(model="gpt-4", temperature=0)
        self.embedding = OpenAIEmbedding(model=embedding_model)
        self.vector_store = SimpleVectorStore()
    
    def process_filing(self, filing: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single filing.
        
        Args:
            filing: Filing dictionary with content and metadata
            
        Returns:
            Processed filing with chunks and embeddings
        """
        # Create document
        doc = Document(
            text=filing["content"],
            metadata={
                "accession_number": filing["accession_number"],
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "company": filing["company"],
                "ticker": filing["ticker"]
            }
        )
        
        # Create chunks
        chunks = self._create_chunks(doc)
        
        # Create embeddings
        embeddings = self._create_embeddings(chunks)
        
        return {
            **filing,
            "chunks": chunks,
            "embeddings": embeddings
        }
    
    def _create_chunks(self, doc: Document) -> List[Dict[str, Any]]:
        """Create text chunks from document."""
        # Simple chunking implementation
        # In production, use more sophisticated chunking
        text = doc.text
        chunks = []
        
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            chunks.append({
                "text": chunk,
                "metadata": doc.metadata,
                "start": start,
                "end": end
            })
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def _create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Create embeddings for chunks."""
        texts = [chunk["text"] for chunk in chunks]
        return self.embedding.get_text_embedding(texts)
    
    def store_in_vector_db(self, processed_filing: Dict[str, Any]):
        """Store processed filing in vector database."""
        # Store chunks and embeddings
        for chunk, embedding in zip(
            processed_filing["chunks"],
            processed_filing["embeddings"]
        ):
            self.vector_store.add(
                text=chunk["text"],
                embedding=embedding,
                metadata=chunk["metadata"]
            ) 