import os

import numpy as np
import pytest

from sec_filing_analyzer.embeddings.embeddings import EmbeddingGenerator


@pytest.fixture
def embedding_generator():
    """Fixture to create an EmbeddingGenerator instance for testing."""
    return EmbeddingGenerator()


def test_initialization():
    """Test the initialization of EmbeddingGenerator."""
    # Test with default model
    generator = EmbeddingGenerator()
    assert generator.dimensions == 1536

    # Test with custom model
    generator = EmbeddingGenerator(model="text-embedding-3-small")
    assert generator.dimensions == 1536


def test_initialization_without_api_key(monkeypatch):
    """Test initialization fails when OPENAI_API_KEY is not set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
        EmbeddingGenerator()


def test_generate_embeddings(embedding_generator):
    """Test generating embeddings for a list of texts."""
    texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = embedding_generator.generate_embeddings(texts)

    # Check shape
    assert embeddings.shape == (2, 1536)

    # Check type
    assert isinstance(embeddings, np.ndarray)

    # Check values are not zero (assuming API call succeeds)
    assert not np.allclose(embeddings, 0)


def test_generate_embeddings_batch(embedding_generator):
    """Test generating embeddings with batch processing."""
    # Create a list of 150 texts to test batch processing
    texts = [f"Test sentence {i}." for i in range(150)]
    embeddings = embedding_generator.generate_embeddings(texts, batch_size=100)

    # Check shape
    assert embeddings.shape == (150, 1536)

    # Check type
    assert isinstance(embeddings, np.ndarray)

    # Check values are not zero (assuming API call succeeds)
    assert not np.allclose(embeddings, 0)


def test_get_embedding_dimensions(embedding_generator):
    """Test getting embedding dimensions."""
    dimensions = embedding_generator.get_embedding_dimensions()
    assert dimensions == 1536


def test_empty_text_list(embedding_generator):
    """Test handling of empty text list."""
    embeddings = embedding_generator.generate_embeddings([])
    assert embeddings.shape == (0, 1536)


def test_embedding_consistency(embedding_generator):
    """Test that the same text produces consistent embeddings."""
    text = "This is a test sentence for consistency."
    embeddings1 = embedding_generator.generate_embeddings([text])
    embeddings2 = embedding_generator.generate_embeddings([text])

    # Check that embeddings are similar enough
    # Using a tolerance of 1e-3 which is appropriate for this model
    assert np.allclose(embeddings1, embeddings2, rtol=1e-3, atol=1e-3)

    # Also check that the embeddings are normalized (length ≈ 1)
    norm1 = np.linalg.norm(embeddings1)
    norm2 = np.linalg.norm(embeddings2)
    assert abs(norm1 - 1.0) < 1e-3
    assert abs(norm2 - 1.0) < 1e-3
