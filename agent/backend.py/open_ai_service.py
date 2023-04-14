from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient

from agent.utils.configuration import load_config

# loguru make file logger
logger.add("logs/{time}.log")


@load_config(location="config/db.yml")
def initialize_db_connection(cfg: DictConfig, open_ai_token: str) -> QdrantClient:
    embeddings = OpenAIEmbeddings(openai_api_key=open_ai_token)
    client = QdrantClient(cfg.db.url, port=cfg.db.port, prefer_grpc=True)
    qdrant = Qdrant(client=client, collection_name="open_ai_db", embedding_function=embeddings.embed_query)

    return qdrant


def embedd_new_documents():
    pass


def search_db():
    pass


if __name__ == "__main__":
    initialize_db_connection(open_ai_token="asdf")
    # embedd_new_documents()
    # search_db()
