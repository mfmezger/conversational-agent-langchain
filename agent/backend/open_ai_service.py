"""This script is used to initialize the chroma db backend with Azure OpenAI."""
import os
from typing import List, Tuple

import weaviate
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from loguru import logger
from omegaconf import DictConfig

from agent.utils.configuration import load_config

load_dotenv()


@load_config(location="config/db/weaviate.yaml")
def get_db_connection(cfg: DictConfig, open_ai_token: str) -> Weaviate:
    """get_db_connection initializes the connection to the chroma db.

    :param cfg: OmegaConf configuration
    :type cfg: DictConfig
    :param open_ai_token: OpenAI API Token
    :type open_ai_token: str
    :return: Chroma DB connection
    :rtype: Chroma
    """
    resource_owner_config = weaviate.AuthApiKey(api_key=cfg.weav.api_key)

    # Initiate the client with the auth config
    client = weaviate.Client(
        url="http://localhost:8080",
        auth_client_secret=resource_owner_config,
        additional_headers={"user": "dev"},
    )
    embedding = OpenAIEmbeddings(chunk_size=1, openai_api_key=open_ai_token)
    return Weaviate(client=client, index_name=cfg.weav.index_name_openai, embedding=embedding, text_key="text")  # TODO: WTF is text kry?


@load_config(location="config/db/weaviate.yaml")
def embedd_documents_openai(dir: str, open_ai_token: str, cfg: DictConfig) -> None:
    """Embedd_documents embedds the documents in the given directory.

    :param cfg: Configuration from the file
    :type cfg: DictConfig
    :param dir: PDF Directory
    :type dir: str
    :param open_ai_token: OpenAI API Token
    :type open_ai_token: str
    """
    weav_connection = get_db_connection(open_ai_token=open_ai_token)
    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    logger.info(f"Loaded {len(docs)} documents.")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    # vector_db.add_texts(texts=texts, metadatas=metadatas)
    embedding = OpenAIEmbeddings(chunk_size=1, openai_api_key=open_ai_token)
    weav_connection.from_texts(texts=texts, weaviate_url=cfg.weav.url, by_text=False, embedding=embedding, metadatas=metadatas)
    logger.info("SUCCESS: Texts embedded.")


def search_documents_openai(open_ai_token: str, query: str, amount: int) -> List[Tuple[Document, float]]:
    """search_documents searches the documents in the Chroma DB with a specific query.

    :param open_ai_token: OpenAI Token
    :type open_ai_token: str
    :param query: The Question for which documents should be searched.
    :type query: str
    :return: List of Results.
    :rtype: List[Tuple[Document, float]]
    """
    vector_db: Chroma = get_db_connection(open_ai_token=open_ai_token)

    docs = vector_db.similarity_search_with_score(query, k=amount)
    logger.info("SUCCESS: Documents found.")
    return docs


def create_summarization(open_ai_token: str, documents):
    """Generate a summary of the given documents.

    :param open_ai_token: _description_
    :type open_ai_token: str
    :param documents: _description_
    :type documents: _type_
    """
    pass


if __name__ == "__main__":

    token = os.getenv("OPENAI_API_KEY")

    if not token:
        raise ValueError("OPENAI_API_KEY is not set.")

    embedd_documents_openai(dir="data", open_ai_token=token)

    # DOCS = search_documents_openai(open_ai_token="", query="Was ist Vanille?", amount=3)
    # print(DOCS)
