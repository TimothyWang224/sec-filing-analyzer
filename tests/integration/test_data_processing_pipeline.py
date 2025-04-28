"""
Test module for the data processing pipeline.

This module tests the complete data processing pipeline including:
- Chunking
- Embedding generation
- Vector store integration
- Filing processing
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from sec_filing_analyzer.data_processing.chunking import FilingChunker
from sec_filing_analyzer.data_retrieval.filing_processor import FilingProcessor
from sec_filing_analyzer.embeddings.embeddings import EmbeddingGenerator
from sec_filing_analyzer.storage.vector_store import LlamaIndexVectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_text():
    """Provide test SEC filing text."""
    return """
    This is a test SEC filing document.
    It contains multiple paragraphs and sections.

    Section 1: Business Overview
    This section describes the company's business operations.

    Section 2: Risk Factors
    This section outlines various risks associated with the business.

    Section 3: Financial Statements
    This section contains the company's financial information.
    """


@pytest.fixture
def pipeline_components():
    """Initialize and return pipeline components."""
    return {
        "embedding_generator": EmbeddingGenerator(),
        "vector_store": LlamaIndexVectorStore(test_mode=True),
        "chunking_processor": FilingChunker(),
    }


def test_pipeline_processing(test_text, pipeline_components):
    """Test the complete data processing pipeline."""
    try:
        # Get components
        embedding_generator = pipeline_components["embedding_generator"]
        vector_store = pipeline_components["vector_store"]
        chunking_processor = pipeline_components["chunking_processor"]

        # Initialize filing processor
        filing_processor = FilingProcessor(vector_store=vector_store)

        # Process chunks
        chunks = chunking_processor.chunk_full_text(test_text)
        assert len(chunks) > 0, "Should generate at least one chunk"
        logger.info(f"Generated {len(chunks)} chunks")

        # Generate embeddings for chunks
        chunk_embeddings = embedding_generator.generate_embeddings(chunks)
        assert len(chunk_embeddings) == len(chunks), "Should generate embedding for each chunk"
        logger.info(f"Generated embeddings for {len(chunk_embeddings)} chunks")

        # Generate embedding for full document
        doc_embedding = embedding_generator.generate_embeddings([test_text])[0]
        assert len(doc_embedding) > 0, "Should generate document embedding"
        logger.info("Generated embedding for full document")

        # Create filing data
        filing_data = {
            "id": "test-filing-001",
            "text": test_text,
            "embedding": doc_embedding,
            "metadata": {"company": "Test Corp", "filing_type": "10-K", "year": "2024"},
            "chunks": chunks,
            "chunk_embeddings": chunk_embeddings,
            "chunk_texts": chunks,
        }

        # Process filing
        processed_data = filing_processor.process_filing(filing_data)
        assert processed_data is not None, "Should return processed data"
        logger.info("Successfully processed filing")

        # Verify vector store
        search_results = vector_store.search_vectors(query_vector=doc_embedding, top_k=5)
        assert len(search_results) > 0, "Should find similar documents"
        logger.info(f"Found {len(search_results)} similar documents in vector store")

        logger.info("Pipeline test completed successfully")

    except Exception as e:
        logger.error(f"Pipeline test failed: {str(e)}")
        raise
