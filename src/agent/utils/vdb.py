"""Vector Database Utilities."""

import warnings
from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.http.models.models import UpdateResult

from agent.utils.config import Config


@dataclass
class VDBResources:
    """Process-scoped vector database resources owned by the API lifespan."""

    config: Config
    sync_client: QdrantClient
    async_client: AsyncQdrantClient
    sparse_embeddings: FastEmbedSparse
    vector_store_cache: dict[str, QdrantVectorStore] = field(default_factory=dict)
    embeddings_cache: dict[tuple[str, str], Embeddings] = field(default_factory=dict)

    async def close(self) -> None:
        """Close both Qdrant clients."""
        try:
            self.sync_client.close()
        finally:
            await self.async_client.close()


def create_vdb_resources(config: Config) -> VDBResources:
    """Create the vector database resources for one application process."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Api key is used with an insecure connection")
        sync_client = QdrantClient(
            location=config.qdrant_url,
            port=config.qdrant_port,
            api_key=config.qdrant_api_key,
            prefer_grpc=config.qdrant_prefer_grpc,
        )
        async_client = AsyncQdrantClient(
            location=config.qdrant_url,
            port=config.qdrant_port,
            api_key=config.qdrant_api_key,
            prefer_grpc=config.qdrant_prefer_grpc,
        )

    return VDBResources(
        config=config,
        sync_client=sync_client,
        async_client=async_client,
        sparse_embeddings=FastEmbedSparse(model_name="Qdrant/bm25"),
    )


def init_vdb(resources: VDBResources, collection_name: str, embedding: Embeddings) -> QdrantVectorStore:
    """Establish a connection to the Qdrant DB."""
    logger.info(f"USING COLLECTION: {collection_name}")

    # QdrantVectorStore requires the synchronous client even when retrieval is
    # initiated by an async route. Direct async Qdrant operations use async_client.
    vector_db = QdrantVectorStore(
        client=resources.sync_client,
        collection_name=collection_name,
        embedding=embedding,
        sparse_embedding=resources.sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="fast-sparse-bm25",
    )
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db


def get_vector_store(resources: VDBResources, collection_name: str, embedding: Embeddings) -> QdrantVectorStore:
    """Return a cached hybrid vector store for a collection."""
    if collection_name not in resources.vector_store_cache:
        resources.vector_store_cache[collection_name] = init_vdb(
            resources=resources,
            collection_name=collection_name,
            embedding=embedding,
        )
    return resources.vector_store_cache[collection_name]


def initialize_vector_db(client: QdrantClient, collection_name: str, embeddings_size: int) -> None:
    """Initialize a collection using the synchronous Qdrant client."""
    if client.collection_exists(collection_name=collection_name):
        logger.info(f"SUCCESS: Collection {collection_name} already exists.")
    else:
        generate_collection(client=client, collection_name=collection_name, embeddings_size=embeddings_size)


def generate_collection(client: QdrantClient, collection_name: str, embeddings_size: int) -> None:
    """Generate a collection using the synchronous Qdrant client."""
    client.set_sparse_model(embedding_model_name="Qdrant/bm25")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


async def delete_documents_by_source(client: AsyncQdrantClient, collection_name: str, source: str) -> UpdateResult:
    """Delete all documents with a matching source from a collection."""
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


async def initialize_vector_db_async(client: AsyncQdrantClient, collection_name: str, embeddings_size: int) -> None:
    """Initialize a collection using the asynchronous Qdrant client."""
    if await client.collection_exists(collection_name=collection_name):
        logger.info(f"SUCCESS: Collection {collection_name} already exists.")
    else:
        await generate_collection_async(client=client, collection_name=collection_name, embeddings_size=embeddings_size)


async def generate_collection_async(client: AsyncQdrantClient, collection_name: str, embeddings_size: int) -> None:
    """Generate a collection using the asynchronous Qdrant client."""
    client.set_sparse_model(embedding_model_name="Qdrant/bm25")
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


async def initialize_all_vector_dbs(config: Config, client: AsyncQdrantClient) -> None:
    """Initialize all configured vector databases asynchronously."""
    await initialize_vector_db_async(
        client=client,
        collection_name=config.qdrant_collection_name,
        embeddings_size=config.embedding_size,
    )
