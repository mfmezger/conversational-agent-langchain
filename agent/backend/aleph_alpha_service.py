"""The script to initialize the chroma db backend with aleph alpha."""
import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain.vectorstores import Chroma
from loguru import logger
from omegaconf import DictConfig

from agent.utils.configuration import load_config

load_dotenv()


@load_config(location="config/chroma_db.yml")
def get_db_connection(cfg: DictConfig, aleph_alpha_token: str) -> Chroma:
    """get_db_connection initializes the connection to the chroma db.

    :param cfg: Configuration file loaded via OmegaConf.
    :type cfg: DictConfig
    :param aleph_alpha_token: Aleph Alpha API Token.
    :type aleph_alpha_token: str
    :return: Chroma DB connection.
    :rtype: Chroma
    """
    embedding = AlephAlphaAsymmetricSemanticEmbedding(aleph_alpha_api_key=aleph_alpha_token)
    vector_db = Chroma(persist_directory=cfg.chroma.persist_directory_aa, embedding_function=embedding)

    logger.info("SUCCESS: Chroma DB initialized.")

    return vector_db


def embedd_documents_aleph_alpha(dir: str, aleph_alpha_token: str) -> None:
    """embedd_documents embedds the documents in the given directory.

    This method uses the Directory Loader for PDFs and the PyPDFLoader to load the documents.
    The documents are then added to the Chroma DB which embedds them without deleting the old collection.
    :param cfg: Configuration from the file
    :type cfg: DictConfig
    :param dir: PDF Directory
    :type dir: str
    :param aleph_alpha_token: Aleph Alpha API Token
    :type aleph_alpha_token: str
    """
    vector_db = get_db_connection(aleph_alpha_token=aleph_alpha_token)

    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    logger.info(f"Loaded {len(docs)} documents.")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vector_db.add_texts(texts=texts, metadatas=metadatas)
    logger.info("SUCCESS: Texts embedded.")
    vector_db.persist()
    logger.info("SUCCESS: Database Persistent.")


def search_documents_aleph_alpha(aleph_alpha_token: str, query: str) -> List[Tuple[Document, float]]:
    """search_documents takes a query and searchs the Chroma DB for similar documents.

    :param aleph_alpha_token: Aleph Alpha API Token
    :type aleph_alpha_token: str
    :param query: The Query that should be searched for.
    :type query: str
    :return: Multiple Documents
    :rtype: List[Tuple[Document, float]]
    """
    vector_db = get_db_connection(aleph_alpha_token=aleph_alpha_token)

    docs = vector_db.similarity_search_with_score(query, k=3)
    logger.info("SUCCESS: Documents found.")
    return docs


if __name__ == "__main__":
    embedd_documents_aleph_alpha("data", os.getenv("AA_Token"))

    # os.environ["ALEPH_ALPHA_API_KEY"] = os.getenv("AA_Token")

    DOCS = search_documents_aleph_alpha(aleph_alpha_token=os.getenv("AA_Token"), query="Muss ich mein Mietwagen volltanken?")
    print(DOCS)
