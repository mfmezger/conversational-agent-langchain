"""Retriever utils."""

from langchain_cohere import CohereEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from ultra_simple_config import load_config

from agent.utils.config import config


@load_config("config/litellm.yml")
def get_retriever(cfg: DictConfig, k: int = 4, collection_name: str = "default") -> BaseRetriever:
    """Create a Vector Database retriever.

    Returns
    -------
        BaseRetriever: Qdrant + Cohere Embeddings Retriever

    """
    embedding = CohereEmbeddings(model=cfg.embedding.model_name)

    qdrant_client = QdrantClient(
        location=config.qdrant_url,
        port=config.qdrant_port,
        api_key=config.qdrant_api_key,
        prefer_grpc=False,
    )

    vector_db = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embedding)
    return vector_db.as_retriever(search_kwargs={"k": k})
