"""Vector Database Utilities."""

import warnings

from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.http.models.models import UpdateResult

from agent.utils.config import Config

_sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
_vector_store_cache: dict[str, QdrantVectorStore] = {}

settings = Config()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Api key is used with an insecure connection")
    _qdrant_client = QdrantClient(
        location=settings.qdrant_url,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Api key is used with an insecure connection")
    _async_qdrant_client = AsyncQdrantClient(
        location=settings.qdrant_url,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )


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
        client=load_vec_db_conn(),
        collection_name=collection_name,
        embedding=embedding,
        sparse_embedding=_sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="fast-sparse-bm25",
    )
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db


def load_vec_db_conn() -> QdrantClient:
    """Return the module-level synchronous QdrantClient singleton.

    Returns
    -------
        QdrantClient: The shared QdrantClient instance.

    """
    return _qdrant_client


def get_async_qdrant_client() -> AsyncQdrantClient:
    """Return the module-level asynchronous QdrantClient singleton.

    Returns
    -------
        AsyncQdrantClient: The shared AsyncQdrantClient instance.

    """
    return _async_qdrant_client


def get_vector_store(collection_name: str, embedding: Embeddings) -> QdrantVectorStore:
    """Return a cached hybrid vector store for a collection."""
    if collection_name not in _vector_store_cache:
        _vector_store_cache[collection_name] = init_vdb(collection_name=collection_name, embedding=embedding)
    return _vector_store_cache[collection_name]


def initialize_vector_db(collection_name: str, embeddings_size: int) -> None:
    """Initializes the vector db for a given backend.

    Args:
    ----
        collection_name (str): Name of the Collection
        embeddings_size (int): Size of the Embeddings

    """
    client = load_vec_db_conn()
    if client.collection_exists(collection_name=collection_name):
        logger.info(f"SUCCESS: Collection {collection_name} already exists.")
    else:
        generate_collection(collection_name=collection_name, embeddings_size=embeddings_size)


def generate_collection(collection_name: str, embeddings_size: int) -> None:
    """Generate a collection for a given backend.

    Args:
    ----
        collection_name (str): Name of the Collection
        embeddings_size (int): Size of the Embeddings

    """
    client = load_vec_db_conn()
    client.set_sparse_model(embedding_model_name="Qdrant/bm25")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


async def delete_documents_by_source(collection_name: str, source: str) -> UpdateResult:
    """Delete all documents with a matching source from a collection."""
    client = get_async_qdrant_client()
    return await client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source)),
                ],
            )
        ),
    )


async def initialize_vector_db_async(collection_name: str, embeddings_size: int) -> None:
    """Initializes the vector db for a given backend asynchronously.

    Args:
    ----
        collection_name (str): Name of the Collection
        embeddings_size (int): Size of the Embeddings

    """
    client = get_async_qdrant_client()
    if await client.collection_exists(collection_name=collection_name):
        logger.info(f"SUCCESS: Collection {collection_name} already exists.")
    else:
        await generate_collection_async(collection_name=collection_name, embeddings_size=embeddings_size)


async def generate_collection_async(collection_name: str, embeddings_size: int) -> None:
    """Generate a collection for a given backend asynchronously.

    Args:
    ----
        collection_name (str): Name of the Collection
        embeddings_size (int): Size of the Embeddings

    """
    client = get_async_qdrant_client()
    client.set_sparse_model(embedding_model_name="Qdrant/bm25")
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


def initialize_all_vector_dbs(config: Config) -> None:
    """Initializes all vector dbs."""
    initialize_vector_db(collection_name=config.qdrant_collection_name, embeddings_size=config.embedding_size)
