"""The script to initialize the chroma db backend with aleph alpha."""
import os

from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models

from agent.utils.configuration import load_config


@load_config(location="config/db.yml")
def initialize_aleph_alpha_vector_db(cfg: DictConfig):
    """Initializes the Aleph Alpha vector db.

    Args:
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)
    collection_name = "Aleph_Alpha"
    try:
        qdrant_client.get_collection(collection_name=collection_name)
        logger.info("SUCCESS: Collection already exists.")
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE),
        )
        logger.info("SUCCESS: Collection created.")


@load_config(location="config/db.yml")
def initialize_open_ai_vector_db(cfg: DictConfig):
    """Initializes the OpenAI vector db.

    Args:
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)
    collection_name = "OpenAI"
    try:
        qdrant_client.get_collection(collection_name=collection_name)
        logger.info("SUCCESS: Collection already exists.")
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
        logger.info("SUCCESS: Collection created.")


@load_config(location="config/db.yml")
def initialize_gpt4all_vector_db(cfg: DictConfig):
    """Initializes the GPT4ALL vector db.

    Args:
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)
    collection_name = "GPT4ALL"
    try:
        qdrant_client.get_collection(collection_name=collection_name)
        logger.info("SUCCESS: Collection already exists.")
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        logger.info("SUCCESS: Collection created.")
