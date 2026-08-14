"""Retriever utils with cached embeddings and vector stores."""

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from agent.utils.config import config
from agent.utils.embeddings import get_embedding_model
from agent.utils.vdb import get_vector_store

# Cache embeddings - created once per model name
_embeddings_cache: dict[tuple[str, str], Embeddings] = {}


def _get_cached_embedding() -> Embeddings:
    """Get or create cached embeddings for the configured provider."""
    key = (config.embedding_provider, config.embedding_model_name)
    if key not in _embeddings_cache:
        _embeddings_cache[key] = get_embedding_model(config)
    return _embeddings_cache[key]


def get_retriever(k: int = 4, collection_name: str = "default") -> BaseRetriever:
    """Create a Vector Database retriever with hybrid search.

    Uses cached embeddings and vector stores for performance.

    Args:
        k: Number of documents to retrieve.
        collection_name: Name of the collection to search.

    Returns:
        BaseRetriever: Qdrant + Cohere Embeddings Retriever with hybrid search.

    """
    vector_db = get_vector_store(collection_name=collection_name, embedding=_get_cached_embedding())
    return vector_db.as_retriever(search_kwargs={"k": k})
