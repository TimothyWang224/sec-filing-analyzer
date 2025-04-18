# Document Chunking Architecture

This document explains the document chunking architecture in the SEC Filing Analyzer.

## Overview

The SEC Filing Analyzer uses a unified approach to document chunking, with the `FilingChunker` class serving as the primary implementation for all document chunking needs. This class provides specialized handling for SEC filings while also supporting generic document chunking.

## FilingChunker

The `FilingChunker` class in `src/sec_filing_analyzer/data_processing/chunking.py` is the main implementation for document chunking. It provides:

1. **SEC Filing-Specific Chunking**: Uses the Edgar library's `ChunkedDocument` class to intelligently chunk SEC filings based on their structure (sections, headers, tables, etc.).

2. **Generic Document Chunking**: Provides a simple token-based chunking method for any text document, respecting token limits and adding appropriate metadata.

3. **Fallback Mechanisms**: Includes fallback mechanisms for when HTML parsing fails or when chunks exceed token limits.

### Key Methods

- `process_filing(filing_data, filing_content)`: Process a SEC filing and return chunked data.
- `chunk_html_filing(html_content)`: Chunk an HTML filing using EDGAR's ChunkedDocument class.
- `chunk_document(text)`: Generic method to chunk any document into smaller pieces for embedding.
- `chunk_full_text(text)`: Split text into chunks based on paragraphs and sentences.
- `_fallback_text_chunking(content)`: Fallback method for simple text-based chunking when HTML processing fails.

## Backward Compatibility

For backward compatibility, a thin wrapper class called `DocumentChunker` is provided in `src/sec_filing_analyzer/semantic/processing/chunking.py`. This class maintains the same interface as the original `DocumentChunker` but uses the enhanced `FilingChunker` internally.

```python
# Old code using DocumentChunker
from sec_filing_analyzer.semantic.processing.chunking import DocumentChunker
chunker = DocumentChunker()
chunks = chunker.chunk_document("This is a document to chunk...")

# New code using FilingChunker directly
from sec_filing_analyzer.data_processing.chunking import FilingChunker
chunker = FilingChunker()
chunks = chunker.chunk_document("This is a document to chunk...")
```

## Recommendations for New Code

For new code, it's recommended to use the `FilingChunker` class directly, as it provides a unified interface for all document chunking needs. The `DocumentChunker` class is maintained only for backward compatibility.

## Implementation Details

### Token Counting and Splitting

The `FilingChunker` uses OpenAI's `tiktoken` library for token counting and LlamaIndex's `TokenTextSplitter` for token-based splitting. This ensures that chunks respect the token limits of embedding models.

### Metadata

The `FilingChunker` adds rich metadata to chunks, including:

- Token counts
- Chunk indices and total chunk counts
- Item information for SEC filings
- Table detection for SEC filings
- Relationship information for split chunks

This metadata is useful for downstream processing and retrieval.

### Error Handling

The `FilingChunker` includes robust error handling, with fallback mechanisms for when primary chunking methods fail. This ensures that documents can still be processed even if they don't conform to expected formats.

## Conclusion

The unified chunking architecture simplifies the codebase, reduces duplication, and provides a clear, consistent interface for all document chunking needs. The `FilingChunker` class is the primary implementation, with the `DocumentChunker` class maintained for backward compatibility.
