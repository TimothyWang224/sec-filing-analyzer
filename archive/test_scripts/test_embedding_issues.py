"""
Test script to diagnose embedding issues with OpenAI API.
"""

import os
import logging
import json
import time
from typing import List, Dict, Any
import tiktoken
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text."""
    tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
    return len(tokenizer.encode(text))

def test_embedding_direct(text: str) -> Dict[str, Any]:
    """Test embedding generation directly using LlamaIndex."""
    result = {
        "success": False,
        "error": None,
        "token_count": count_tokens(text),
        "text_length": len(text),
        "embedding_length": None
    }
    
    try:
        # Initialize embedding model
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Generate embedding
        embedding = embed_model.get_text_embedding(text)
        
        # Update result
        result["success"] = True
        result["embedding_length"] = len(embedding)
        
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
    
    return result

def test_batch_embedding(texts: List[str], batch_size: int = 10) -> Dict[str, Any]:
    """Test batch embedding generation."""
    result = {
        "success": False,
        "error": None,
        "total_texts": len(texts),
        "batch_size": batch_size,
        "token_counts": [count_tokens(text) for text in texts],
        "text_lengths": [len(text) for text in texts],
        "successful_embeddings": 0
    }
    
    try:
        # Initialize embedding model
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Ensure batch is properly formatted
            sanitized_batch = [str(text) if text is not None else "" for text in batch]
            
            try:
                # Generate embeddings for batch
                batch_embeddings = embed_model.get_text_embedding_batch(sanitized_batch)
                all_embeddings.extend(batch_embeddings)
                result["successful_embeddings"] += len(batch_embeddings)
                
                # Add delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {e}")
                # Continue with next batch
        
        # Update result
        result["success"] = result["successful_embeddings"] > 0
        
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
    
    return result

def test_large_text_handling() -> Dict[str, Any]:
    """Test handling of large texts that exceed token limits."""
    # Create a text that exceeds the token limit
    large_text = "This is a test. " * 4000  # Should be around 16,000 tokens
    
    # Count tokens
    token_count = count_tokens(large_text)
    
    # Split text into smaller chunks
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(large_text)
    
    # Create chunks of 4000 tokens each
    max_tokens = 4000
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    # Test embedding each chunk
    chunk_results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Testing chunk {i+1}/{len(chunks)}")
        result = test_embedding_direct(chunk)
        chunk_results.append(result)
        time.sleep(1)  # Add delay to avoid rate limiting
    
    return {
        "original_token_count": token_count,
        "num_chunks": len(chunks),
        "chunk_results": chunk_results
    }

def test_input_format_issues() -> Dict[str, Any]:
    """Test various input formats to identify what causes the '$.input' is invalid error."""
    test_cases = [
        {"name": "normal_text", "input": "This is a normal text."},
        {"name": "empty_string", "input": ""},
        {"name": "none_value", "input": None},
        {"name": "special_chars", "input": "Text with special chars: !@#$%^&*()_+"},
        {"name": "unicode", "input": "Unicode text: 你好, こんにちは, 안녕하세요"},
        {"name": "very_long", "input": "Long text. " * 1000},
        {"name": "with_newlines", "input": "Text with\nnewlines\nand\rcarriage\returns"},
        {"name": "with_tabs", "input": "Text with\ttabs\tand spaces"},
        {"name": "with_null_bytes", "input": "Text with\x00null\x00bytes"},
        {"name": "with_control_chars", "input": "Text with control chars: \x01\x02\x03\x04"},
        {"name": "json_string", "input": json.dumps({"key": "value"})},
        {"name": "html", "input": "<html><body><h1>HTML Content</h1></body></html>"},
        {"name": "xml", "input": "<xml><tag>XML Content</tag></xml>"}
    ]
    
    results = {}
    for test_case in test_cases:
        name = test_case["name"]
        input_text = test_case["input"]
        
        logger.info(f"Testing input format: {name}")
        
        # Handle None value specially
        if input_text is None:
            input_text = ""
        
        try:
            result = test_embedding_direct(input_text)
            results[name] = result
        except Exception as e:
            results[name] = {"success": False, "error": f"{type(e).__name__}: {str(e)}"}
        
        time.sleep(0.5)  # Add delay to avoid rate limiting
    
    return results

def main():
    """Run the tests."""
    logger.info("Starting embedding tests")
    
    # Test input format issues
    logger.info("Testing input format issues")
    input_format_results = test_input_format_issues()
    
    # Test large text handling
    logger.info("Testing large text handling")
    large_text_results = test_large_text_handling()
    
    # Test batch embedding
    logger.info("Testing batch embedding")
    texts = ["Text " + str(i) for i in range(30)]
    batch_results_small = test_batch_embedding(texts, batch_size=5)
    batch_results_large = test_batch_embedding(texts, batch_size=20)
    
    # Combine results
    all_results = {
        "input_format_tests": input_format_results,
        "large_text_tests": large_text_results,
        "batch_tests_small": batch_results_small,
        "batch_tests_large": batch_results_large
    }
    
    # Save results to file
    with open("embedding_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("Tests completed. Results saved to embedding_test_results.json")

if __name__ == "__main__":
    main()
