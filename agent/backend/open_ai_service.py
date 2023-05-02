"""This script is used to initialize the chroma db backend with Azure OpenAI."""
import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from loguru import logger
from omegaconf import DictConfig

from utils.configuration import load_config

load_dotenv()
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://openaiendpoint.openai.azure.com/"


@load_config(location="config/chroma_db.yml")
def get_db_connection(cfg: DictConfig, open_ai_token: str) -> Chroma:
    """get_db_connection initializes the connection to the chroma db.

    :param cfg: OmegaConf configuration
    :type cfg: DictConfig
    :param open_ai_token: OpenAI API Token
    :type open_ai_token: str
    :return: Chroma DB connection
    :rtype: Chroma
    """
    embedding = OpenAIEmbeddings(chunk_size=1, openai_api_key=open_ai_token)
    vector_db = Chroma(persist_directory=cfg.chroma.persist_directory, embedding_function=embedding)
    logger.info("SUCCESS: Chroma DB initialized.")
    return vector_db


def embedd_documents_openai(dir: str, open_ai_token: str) -> None:
    """embedd_documents embedds the documents in the given directory.

    :param cfg: Configuration from the file
    :type cfg: DictConfig
    :param dir: PDF Directory
    :type dir: str
    :param open_ai_token: OpenAI API Token
    :type open_ai_token: str
    """
    vector_db: Chroma = get_db_connection(open_ai_token=open_ai_token)

    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    logger.info(f"Loaded {len(docs)} documents.")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vector_db.add_texts(texts=texts, metadatas=metadatas)
    logger.info("SUCCESS: Texts embedded.")
    vector_db.persist()
    logger.info("SUCCESS: Database Persistent.")


def search_documents_openai(open_ai_token: str, query: str) -> List[Tuple[Document, float]]:
    """search_documents searches the documents in the Chroma DB with a specific query.

    :param open_ai_token: OpenAI Token
    :type open_ai_token: str
    :param query: The Question for which documents should be searched.
    :type query: str
    :return: List of Results.
    :rtype: List[Tuple[Document, float]]
    """
    vector_db: Chroma = get_db_connection(open_ai_token=open_ai_token)

    docs = vector_db.similarity_search_with_score(query, k=3)
    logger.info("SUCCESS: Documents found.")
    return docs


if __name__ == "__main__":

    embedd_documents_openai("data", os.getenv("OPENAI_API_KEY"))

    DOCS = search_documents_openai(open_ai_token="", query="Muss ich mein Mietwagen volltanken?")
    print(DOCS)
