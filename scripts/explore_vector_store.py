"""
Utility script to explore the vector store contents.

This script allows you to:
1. List all documents in the vector store
2. View specific documents by ID
3. Search for similar documents
"""

import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from sec_filing_analyzer.storage import LlamaIndexVectorStore
from sec_filing_analyzer.config import StorageConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def list_documents(vector_store: LlamaIndexVectorStore, limit: int = 10, offset: int = 0,
                  ticker: Optional[str] = None, filing_type: Optional[str] = None) -> None:
    """
    List documents in the vector store.

    Args:
        vector_store: The vector store to query
        limit: Maximum number of documents to return
        offset: Number of documents to skip
        ticker: Filter by ticker symbol
        filing_type: Filter by filing type
    """
    # Get all document IDs from the vector store
    all_ids = vector_store.list_documents()

    # Apply filters if specified
    filtered_ids = all_ids
    if ticker or filing_type:
        filtered_ids = []
        for doc_id in all_ids:
            metadata = vector_store.get_document_metadata(doc_id)
            if metadata:
                if ticker and metadata.get('ticker') != ticker:
                    continue
                if filing_type and metadata.get('form') != filing_type:
                    continue
                filtered_ids.append(doc_id)

    # Apply pagination
    paginated_ids = filtered_ids[offset:offset+limit]

    # Print document information
    print(f"Found {len(filtered_ids)} documents (showing {offset+1}-{min(offset+limit, len(filtered_ids))})")
    print("-" * 80)

    for i, doc_id in enumerate(paginated_ids, start=1):
        metadata = vector_store.get_document_metadata(doc_id)
        text = vector_store.get_document_text(doc_id)

        print(f"Document {offset+i}:")
        print(f"  ID: {doc_id}")
        if metadata:
            print(f"  Ticker: {metadata.get('ticker', 'N/A')}")
            print(f"  Form: {metadata.get('form', 'N/A')}")
            print(f"  Filing Date: {metadata.get('filing_date', 'N/A')}")
            print(f"  Item: {metadata.get('item', 'N/A')}")
            print(f"  Is Table: {metadata.get('is_table', False)}")

        # Print a snippet of the text
        if text:
            snippet = text[:200] + "..." if len(text) > 200 else text
            print(f"  Text Snippet: {snippet}")

        print("-" * 80)

def view_document(vector_store: LlamaIndexVectorStore, doc_id: str) -> None:
    """
    View a specific document by ID.

    Args:
        vector_store: The vector store to query
        doc_id: The document ID to view
    """
    metadata = vector_store.get_document_metadata(doc_id)
    text = vector_store.get_document_text(doc_id)
    embedding = vector_store.get_document_embedding(doc_id)

    print(f"Document ID: {doc_id}")
    print("-" * 80)

    if metadata:
        print("Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    if embedding:
        print("\nEmbedding:")
        print(f"  Dimensions: {len(embedding)}")
        # Format the first 5 values with limited precision
        first_values = [f"{val:.6f}" for val in embedding[:5]]
        print(f"  First 5 values: {first_values}...")

    if text:
        print("\nText:")
        print(text)
    else:
        print("\nNo text available for this document.")

def search_documents(vector_store: LlamaIndexVectorStore, query: str, top_k: int = 5) -> None:
    """
    Search for documents similar to the query.

    Args:
        vector_store: The vector store to query
        query: The search query
        top_k: Number of results to return
    """
    import numpy as np
    from sec_filing_analyzer.embeddings import EmbeddingGenerator

    def cosine_similarity(a, b):
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Generate embedding for the query
    embedding_generator = EmbeddingGenerator()
    query_embedding = embedding_generator.generate_embeddings([query])[0]

    # Load all document embeddings
    all_ids = vector_store.list_documents()

    # Calculate similarity scores
    scores = {}
    for doc_id in all_ids:
        embedding = vector_store.get_document_embedding(doc_id)
        if embedding is not None:
            try:
                score = cosine_similarity(query_embedding, embedding)
                if not np.isnan(score):  # Skip NaN scores
                    scores[doc_id] = score
            except Exception as e:
                print(f"Error calculating similarity for {doc_id}: {e}")

    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Get top k results
    top_results = sorted_scores[:top_k]

    # Format the results for display
    formatted_results = []
    for doc_id, score in top_results:
        metadata = vector_store.get_document_metadata(doc_id)
        text = vector_store.get_document_text(doc_id)

        formatted_results.append({
            'id': doc_id,
            'score': score,
            'metadata': metadata,
            'text': text
        })

    print(f"Search results for: '{query}'")
    print("-" * 80)

    if not formatted_results:
        print("No results found.")
        return

    for i, result in enumerate(formatted_results, start=1):
        doc_id = result['id']
        score = result['score']
        metadata = result['metadata']
        text = result['text']

        print(f"Result {i} (Score: {score:.4f}):")
        print(f"  ID: {doc_id}")

        if metadata:
            print(f"  Ticker: {metadata.get('ticker', 'N/A')}")
            print(f"  Form: {metadata.get('form', 'N/A')}")
            print(f"  Filing Date: {metadata.get('filing_date', 'N/A')}")
            item = metadata.get('item', 'N/A')
            if isinstance(metadata.get('chunk_metadata'), dict):
                item = metadata['chunk_metadata'].get('item', item)
            print(f"  Item: {item}")

        # Print a snippet of the text
        if text:
            snippet = text[:200] + "..." if len(text) > 200 else text
            print(f"  Text Snippet: {snippet}")

        print("-" * 80)

def export_documents(vector_store: LlamaIndexVectorStore, output_file: str,
                    include_embeddings: bool = False, truncate_embeddings: bool = True) -> None:
    """
    Export documents from the vector store to a JSON file.

    Args:
        vector_store: The vector store to query
        output_file: Path to the output file
        include_embeddings: Whether to include embeddings in the export
        truncate_embeddings: Whether to truncate embeddings to first 10 dimensions
    """
    # Get all document IDs from the vector store
    all_ids = vector_store.list_documents()

    # Collect document data
    documents = []
    for doc_id in all_ids:
        doc_data = {
            'id': doc_id,
            'metadata': vector_store.get_document_metadata(doc_id),
            'text': vector_store.get_document_text(doc_id)
        }

        if include_embeddings:
            embedding = vector_store.get_document_embedding(doc_id)
            if embedding and truncate_embeddings:
                # Only include first 10 dimensions to keep file size manageable
                doc_data['embedding'] = embedding[:10]
                doc_data['embedding_truncated'] = True
                doc_data['embedding_dimensions'] = len(embedding)
            else:
                doc_data['embedding'] = embedding

        documents.append(doc_data)

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(documents)} documents to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Explore vector store contents')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # List documents command
    list_parser = subparsers.add_parser('list', help='List documents in the vector store')
    list_parser.add_argument('--limit', type=int, default=10, help='Maximum number of documents to return')
    list_parser.add_argument('--offset', type=int, default=0, help='Number of documents to skip')
    list_parser.add_argument('--ticker', type=str, help='Filter by ticker symbol')
    list_parser.add_argument('--filing-type', type=str, help='Filter by filing type')

    # View document command
    view_parser = subparsers.add_parser('view', help='View a specific document')
    view_parser.add_argument('doc_id', type=str, help='Document ID to view')

    # Search documents command
    search_parser = subparsers.add_parser('search', help='Search for documents')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')

    # Export documents command
    export_parser = subparsers.add_parser('export', help='Export documents to a file')
    export_parser.add_argument('output_file', type=str, help='Path to the output file')
    export_parser.add_argument('--include-embeddings', action='store_true', help='Include embeddings in the export')
    export_parser.add_argument('--full-embeddings', action='store_true', help='Include full embeddings instead of truncated ones')

    args = parser.parse_args()

    # Initialize vector store
    vector_store = LlamaIndexVectorStore(
        store_path=StorageConfig().vector_store_path
    )

    # Run the appropriate command
    if args.command == 'list':
        list_documents(
            vector_store=vector_store,
            limit=args.limit,
            offset=args.offset,
            ticker=args.ticker,
            filing_type=args.filing_type
        )
    elif args.command == 'view':
        view_document(
            vector_store=vector_store,
            doc_id=args.doc_id
        )
    elif args.command == 'search':
        search_documents(
            vector_store=vector_store,
            query=args.query,
            top_k=args.top_k
        )
    elif args.command == 'export':
        export_documents(
            vector_store=vector_store,
            output_file=args.output_file,
            include_embeddings=args.include_embeddings,
            truncate_embeddings=not args.full_embeddings
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
