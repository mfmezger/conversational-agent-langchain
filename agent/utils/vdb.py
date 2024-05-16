"""Vector Database Utilities."""
import os

from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.embeddings import Embeddings
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
def load_vec_db_conn(cfg: DictConfig) -> QdrantClient:
    """Load the Vector Database Connection."""
    return QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc), cfg


def initialize_aleph_alpha_vector_db() -> None:
    """Initializes the Aleph Alpha vector db."""
    qdrant_client, cfg = load_vec_db_conn()
    try:
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_aa)
        logger.info(f"SUCCESS: Collection {cfg.qdrant.collection_name_aa} already exists.")
    except UnexpectedResponse:
        generate_collection_aleph_alpha(qdrant_client, collection_name=cfg.qdrant.collection_name_aa, embeddings_size=cfg.aleph_alpha_embeddings.size)


def generate_collection_aleph_alpha(qdrant_client: Qdrant, collection_name: str, embeddings_size: int) -> None:
    """Generate a collection for the Aleph Alpha Backend.

    Args:
    ----
        qdrant_client (_type_): _description_
        collection_name (_type_): _description_
        embeddings_size (_type_): _description_
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


def initialize_open_ai_vector_db() -> None:
    """Initializes the OpenAI vector db.

    Args:
    ----
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client, cfg = load_vec_db_conn()

    try:
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_openai)
        logger.info(f"SUCCESS: Collection {cfg.qdrant.collection_name_openai} already exists.")
    except UnexpectedResponse:
        generate_collection_openai(qdrant_client, collection_name=cfg.qdrant.collection_name_openai)


def generate_collection_openai(qdrant_client: Qdrant, collection_name: str) -> None:
    """Generate a collection for the OpenAI Backend.

    Args:
    ----
        qdrant_client (_type_): Qdrant Client Langchain.
        collection_name (_type_): Name of the Collection
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


def initialize_gpt4all_vector_db() -> None:
    """Initializes the GPT4ALL vector db.

    Args:
    ----
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client, cfg = load_vec_db_conn()

    try:
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_gpt4all)
        logger.info(f"SUCCESS: Collection {cfg.qdrant.collection_name_gpt4all} already exists.")
    except UnexpectedResponse:
        generate_collection_gpt4all(qdrant_client, collection_name=cfg.qdrant.collection_name_gpt4all)


def generate_collection_gpt4all(qdrant_client: Qdrant, collection_name: str) -> None:
    """Generate a collection for the GPT4ALL Backend.

    Args:
    ----
        qdrant_client (Qdrant): Qdrant Client
        collection_name (str): Name of the Collection
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")
