"""Vector Database Utilities."""

import warnings

from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from loguru import logger
from qdrant_client import QdrantClient, models

from agent.utils.config import Config

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

settings = Config()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Api key is used with an insecure connection")
    qdrant_client = QdrantClient(
        location=settings.qdrant_url,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        prefer_grpc=settings.qdrant_prefer_http,
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
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embedding,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="fast-sparse-bm25",
    )
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db


def load_vec_db_conn() -> QdrantClient:
    """Load the Vector Database Connection.

    This function creates a new QdrantClient instance using the configuration
    parameters provided in the 'cfg' argument. The QdrantClient is used to
    interact with the Qdrant vector database.

    Returns
    -------
        Tuple[QdrantClient, DictConfig]: A tuple containing the created
                                        QdrantClient instance and the
                                        original configuration object.

    """
    return qdrant_client


def initialize_vector_db(collection_name: str, embeddings_size: int) -> None:
    """Initializes the vector db for a given backend.

    Args:
    ----
        collection_name (str): Name of the Collection
        embeddings_size (int): Size of the Embeddings

    """
    qdrant_client = load_vec_db_conn()

    if qdrant_client.collection_exists(collection_name=collection_name):
        logger.info(f"SUCCESS: Collection {collection_name} already exists.")
    else:
        generate_collection(collection_name=collection_name, embeddings_size=embeddings_size)


def generate_collection(collection_name: str, embeddings_size: int) -> None:
    """Generate a collection for a given backend.

    Args:
    ----
        qdrant_client (Qdrant): Qdrant Client
        collection_name (str): Name of the Collection
        embeddings_size (int): Size of the Embeddings

    """
    qdrant_client = load_vec_db_conn()
    qdrant_client.set_sparse_model(embedding_model_name="Qdrant/bm25")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
        sparse_vectors_config=qdrant_client.get_fastembed_sparse_vector_params(),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


def initialize_all_vector_dbs(config: Config) -> None:
    """Initializes all vector dbs."""
    initialize_vector_db(collection_name=config.qdrant_collection_name, embeddings_size=config.embedding_size)
