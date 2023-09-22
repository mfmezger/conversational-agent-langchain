"""The script to initialize the chroma db backend with aleph alpha."""
import os
from typing import List

from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models

from agent.utils.configuration import load_config

load_dotenv()


@load_config(location="config/db.yml")
def initialize_aleph_alpha_vector_db(cfg: DictConfig) -> None:
    """Initializes the Aleph Alpha vector db.

    Args:
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)
    collection_name = "Aleph_Alpha"
    try:
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_aa)
        logger.info(f"SUCCESS: Collection {collection_name} already exists.")
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=cfg.qdrant.collection_name_aa,
            vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE),
        )
        logger.info(f"SUCCESS: Collection {collection_name} created.")


@load_config(location="config/db.yml")
def setup_connection_vector_db(cfg: DictConfig) -> Qdrant:
    """Sets up the connection to the vector db.

    Args:
        cfg (DictConfig): Configuration from the file

    Returns:
        Qdrant: The vector db
    """
    embedding = AlephAlphaAsymmetricSemanticEmbedding(aleph_alpha_api_key=aleph_alpha_token)  # type: ignore
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)

    vector_db = Qdrant(client=qdrant_client, collection_name=cfg.qdrant.collection_name_aa, embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db


def parse_txts(text: str, file_name: str, seperator: str, vector_db: Qdrant) -> None:
    """Parse the texts and add them to the vector db.

    Text should be marked with a link then </LINK> and then the text.

    Args:
        text (str): The text to parse
        file_name (str): Name of the file
        seperator (str): The seperator to split the text at
        vector_db (Qdrant): The vector db
    """
    # split every text in two parts one before </LINK> and one after
    link = text.split("</LINK>")[0]
    text = text.split("</LINK>")[1]

    # split the text at the seperator
    text_list: List = text.split(seperator)

    # check if first and last element are empty
    if not text_list[0]:
        text_list.pop(0)
    if not text_list[-1]:
        text_list.pop(-1)

    # meta data is a list of dicts including the "file_name" and the "link"
    metadata_list: List = []
    for i in range(len(text_list)):
        metadata_list.append({"file_name": file_name, "link": link})

    vector_db.add_texts(texts=text_list, metadatas=metadata_list)


def parse_pdf(dir: str, vector_db: Qdrant) -> None:
    """Parse the pdfs and add them to the vector db.

    Args:
        dir (str): The directory to parse
        vector_db (Qdrant): The vector db
    """
    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
    length_function = len
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=2000,
        chunk_overlap=400,
        length_function=length_function,
    )
    docs = loader.load_and_split(splitter)

    logger.info(f"Loaded {len(docs)} documents.")
    text_list = [doc.page_content for doc in docs]
    metadata_list = [doc.metadata for doc in docs]

    vector_db.add_texts(texts=text_list, metadatas=metadata_list)

    logger.info("SUCCESS: Texts added to Qdrant DB.")


if __name__ == "__main__":
    initialize_aleph_alpha_vector_db()
    vector_db = setup_connection_vector_db()

    parse_pdf(dir="data/", vector_db=vector_db)
    # iterate over everything in the data/txt folder
    for file_name in os.listdir("data/txt"):
        with open(f"data/txt/{file_name}") as f:
            text = f.read()
            parse_txts(text=text, file_name=file_name, seperator="###", vector_db=vector_db)
