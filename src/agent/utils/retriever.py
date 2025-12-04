"""Retriever utils with cached embeddings and vector stores."""

from langchain_cohere import CohereEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from agent.utils.config import config
from agent.utils.vdb import qdrant_client, sparse_embeddings

# Cache embeddings - created once per model name
_embeddings_cache: dict[str, CohereEmbeddings] = {}


def _get_cached_embedding() -> CohereEmbeddings:
    """Get or create cached Cohere embeddings."""
    model_name = config.embedding_model_name
    if model_name not in _embeddings_cache:
        _embeddings_cache[model_name] = CohereEmbeddings(model=model_name)
    return _embeddings_cache[model_name]


# Cache vector stores per collection
_vector_store_cache: dict[str, QdrantVectorStore] = {}


def _get_cached_vector_store(collection_name: str) -> QdrantVectorStore:
    """Get or create cached QdrantVectorStore for a collection."""
    if collection_name not in _vector_store_cache:
        _vector_store_cache[collection_name] = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=_get_cached_embedding(),
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="fast-sparse-bm25",
        )
    return _vector_store_cache[collection_name]


def get_retriever(k: int = 4, collection_name: str = "default") -> BaseRetriever:
    """Create a Vector Database retriever with hybrid search.

    Uses cached embeddings and vector stores for performance.

    Args:
        k: Number of documents to retrieve.
        collection_name: Name of the collection to search.

    Returns:
        BaseRetriever: Qdrant + Cohere Embeddings Retriever with hybrid search.

    """
    vector_db = _get_cached_vector_store(collection_name)
    return vector_db.as_retriever(search_kwargs={"k": k})
