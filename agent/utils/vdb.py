"""Vector Database Utilities."""

import os

from langchain_core.embeddings import Embeddings
from langchain_qdrant import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from ultra_simple_config import load_config


def init_vdb(cfg: DictConfig, collection_name: str, embedding: Embeddings) -> Qdrant:
    """Establish a connection to the Qdrant DB.

    Args:
    ----
        cfg (DictConfig): the configuration from the file.
        collection_name (str): name of the collection in the Qdrant DB.
        embedding (Embeddings): Embedding Type.

    Returns:
    -------
        Qdrant: Established Connection to the Vector DB including Embeddings.

    """
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)

    logger.info(f"USING COLLECTION: {collection_name}")

    vector_db = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db


@load_config("config/main.yml")
def load_vec_db_conn(cfg: DictConfig) -> tuple[QdrantClient, DictConfig]:
    """Load the Vector Database Connection.

    This function creates a new QdrantClient instance using the configuration
    parameters provided in the 'cfg' argument. The QdrantClient is used to
    interact with the Qdrant vector database.

    Args:
    ----
        cfg (DictConfig): A configuration object that contains the settings
                          for the QdrantClient. It should have 'url', 'port',
                          and 'prefer_grpc' fields under the 'qdrant' key.

    Returns:
    -------
        Tuple[QdrantClient, DictConfig]: A tuple containing the created
                                         QdrantClient instance and the
                                         original configuration object.

    """
    return (QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc), cfg)


def initialize_vector_db(collection_name: str, embeddings_size: int) -> None:
    """Initializes the vector db for a given backend.

    Args:
    ----
        collection_name (str): Name of the Collection
        embeddings_size (int): Size of the Embeddings

    """
    qdrant_client, _ = load_vec_db_conn()

    try:
        qdrant_client.get_collection(collection_name=collection_name)
        logger.info(f"SUCCESS: Collection {collection_name} already exists.")
    except UnexpectedResponse:
        generate_collection(collection_name=collection_name, embeddings_size=embeddings_size)


def generate_collection(collection_name: str, embeddings_size: int) -> None:
    """Generate a collection for a given backend.

    Args:
    ----
        qdrant_client (Qdrant): Qdrant Client
        collection_name (str): Name of the Collection
        embeddings_size (int): Size of the Embeddings

    """
    qdrant_client, _ = load_vec_db_conn()
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


@load_config("config/main.yml")
def initialize_all_vector_dbs(cfg: DictConfig) -> None:
    """Initializes all vector dbs."""
    initialize_vector_db(cfg.qdrant.collection_name_aa, cfg.aleph_alpha_embeddings.size)
    initialize_vector_db(cfg.qdrant.collection_name_openai, cfg.openai_embeddings.size)
    initialize_vector_db(cfg.qdrant.collection_name_gpt4all, 2048)
    initialize_vector_db(cfg.qdrant.collection_name_cohere, cfg.cohere_embeddings.size)
    initialize_vector_db(cfg.qdrant.collection_name_ollama, cfg.ollama_embeddings.size)
