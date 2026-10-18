"""Retriever utilities using application-owned embeddings and vector stores."""

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from agent.utils.embeddings import get_embedding_model
from agent.utils.vdb import VDBResources, get_vector_store


def _get_cached_embedding(resources: VDBResources) -> Embeddings:
    """Get or create cached embeddings for the configured provider."""
    config = resources.config
    key = (config.embedding_provider, config.embedding_model_name)
    if key not in resources.embeddings_cache:
        resources.embeddings_cache[key] = get_embedding_model(config)
    return resources.embeddings_cache[key]


def get_retriever(resources: VDBResources, k: int = 4, collection_name: str = "default") -> BaseRetriever:
    """Create a Qdrant hybrid-search retriever."""
    vector_db = get_vector_store(
        resources=resources,
        collection_name=collection_name,
        embedding=_get_cached_embedding(resources),
    )
    return vector_db.as_retriever(search_kwargs={"k": k})
