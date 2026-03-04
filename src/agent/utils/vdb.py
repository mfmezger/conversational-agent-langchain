"""Vector Database Utilities."""

import warnings

from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from agent.utils.config import Config

settings = Config()

_qdrant_client: QdrantClient | None = None
_async_qdrant_client: AsyncQdrantClient | None = None
_sparse_embeddings: FastEmbedSparse | None = None

def get_qdrant_client() -> QdrantClient:
    """Get or initialize the Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Api key is used with an insecure connection")
            _qdrant_client = QdrantClient(
                location=settings.qdrant_url,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                prefer_grpc=settings.qdrant_prefer_http,
            )
    return _qdrant_client

def get_async_qdrant_client() -> AsyncQdrantClient:
    """Get or initialize the async Qdrant client."""
    global _async_qdrant_client
    if _async_qdrant_client is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Api key is used with an insecure connection")
            _async_qdrant_client = AsyncQdrantClient(
                location=settings.qdrant_url,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                prefer_grpc=settings.qdrant_prefer_http,
            )
    return _async_qdrant_client

def get_sparse_embeddings() -> FastEmbedSparse:
    """Get or initialize the sparse embeddings model."""
    global _sparse_embeddings
    if _sparse_embeddings is None:
        _sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    return _sparse_embeddings

def init_vdb(collection_name: str, embedding: Embeddings) -> QdrantVectorStore:
    """Establish a connection to the Qdrant DB.

    Args:
    ----
        collection_name (str): name of the collection in the Qdrant DB.
        embedding (Embeddings): Embedding Type.

    Returns:
    -------
        Qdrant: Established Connection to the Vector DB including Embeddings.

    """
    logger.info(f"USING COLLECTION: {collection_name}")

    vector_db = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=collection_name,
        embedding=embedding,
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="fast-sparse-bm25",
    )
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db


def initialize_vector_db(collection_name: str, embeddings_size: int) -> None:
    """Initializes the vector db for a given backend.

    Args:
    ----
        collection_name (str): Name of the Collection
        embeddings_size (int): Size of the Embeddings

    """
    qdrant_client = get_qdrant_client()

    if qdrant_client.collection_exists(collection_name=collection_name):
        logger.info(f"SUCCESS: Collection {collection_name} already exists.")
    else:
        generate_collection(collection_name=collection_name, embeddings_size=embeddings_size)

async def initialize_vector_db_async(collection_name: str, embeddings_size: int) -> None:
    """Initializes the vector db asynchronously for a given backend.

    Args:
    ----
        collection_name (str): Name of the Collection
        embeddings_size (int): Size of the Embeddings

    """
    async_qdrant_client = get_async_qdrant_client()

    if await async_qdrant_client.collection_exists(collection_name=collection_name):
        logger.info(f"SUCCESS: Collection {collection_name} already exists.")
    else:
        await generate_collection_async(collection_name=collection_name, embeddings_size=embeddings_size)


def generate_collection(collection_name: str, embeddings_size: int) -> None:
    """Generate a collection for a given backend.

    Args:
    ----
        collection_name (str): Name of the Collection
        embeddings_size (int): Size of the Embeddings

    """
    qdrant_client = get_qdrant_client()
    qdrant_client.set_sparse_model(embedding_model_name="Qdrant/bm25")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
        sparse_vectors_config=qdrant_client.get_fastembed_sparse_vector_params(),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")

async def generate_collection_async(collection_name: str, embeddings_size: int) -> None:
    """Generate a collection asynchronously for a given backend.

    Args:
    ----
        collection_name (str): Name of the Collection
        embeddings_size (int): Size of the Embeddings

    """
    async_qdrant_client = get_async_qdrant_client()
    # Note: set_sparse_model is synchronous and modifies client state
    async_qdrant_client.set_sparse_model(embedding_model_name="Qdrant/bm25")
    await async_qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
        sparse_vectors_config=async_qdrant_client.get_fastembed_sparse_vector_params(),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


def initialize_all_vector_dbs(config: Config) -> None:
    """Initializes all vector dbs."""
    initialize_vector_db(collection_name=config.qdrant_collection_name, embeddings_size=config.embedding_size)
