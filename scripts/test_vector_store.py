"""
Test script for the vector store.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from sec_filing_analyzer.config import StorageConfig
from sec_filing_analyzer.storage import LlamaIndexVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Initialize vector store
    vector_store = LlamaIndexVectorStore(store_path=StorageConfig().vector_store_path)

    # Create a test document
    doc_id = "test_document"
    text = "This is a test document for the vector store."
    embedding = [0.1] * 1536  # Create a dummy embedding
    metadata = {"title": "Test Document", "author": "Test Author", "date": "2023-01-01"}

    # Add the document to the vector store
    vector_store.upsert_vectors(vectors=[embedding], ids=[doc_id], metadata=[metadata], texts=[text])

    # List all documents
    all_docs = vector_store.list_documents()
    logger.info(f"All documents: {all_docs}")

    # Get the document
    doc_metadata = vector_store.get_document_metadata(doc_id)
    doc_text = vector_store.get_document_text(doc_id)
    doc_embedding = vector_store.get_document_embedding(doc_id)

    logger.info(f"Document metadata: {doc_metadata}")
    logger.info(f"Document text: {doc_text}")
    logger.info(f"Document embedding length: {len(doc_embedding) if doc_embedding else 0}")

    # Check if files were created
    vector_store_path = Path(StorageConfig().vector_store_path)
    metadata_path = vector_store_path / "metadata" / f"{doc_id}.json"
    text_path = vector_store_path / "text" / f"{doc_id}.txt"
    embedding_path = vector_store_path / "embeddings" / f"{doc_id}.json"

    logger.info(f"Metadata file exists: {metadata_path.exists()}")
    logger.info(f"Text file exists: {text_path.exists()}")
    logger.info(f"Embedding file exists: {embedding_path.exists()}")


if __name__ == "__main__":
    main()
