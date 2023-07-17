"""GPT4ALL Backend Service."""
import os
from typing import List, Tuple

from dotenv import load_dotenv
from gpt4all import GPT4All
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models

from agent.utils.configuration import load_config
from agent.utils.utility import generate_prompt

load_dotenv()

# TODO: do you need to preload the model?


qdrant_client = QdrantClient("http://localhost", port=6333, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=False)
collection_name = "GPT4ALL"
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
def get_db_connection(cfg: DictConfig) -> Qdrant:
    """Initializes a connection to the Chroma DB.

    Args:
        cfg (DictConfig): The configuration file loaded via OmegaConf.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
        Chroma: The Chroma DB connection.
    """
    embedding = GPT4AllEmbeddings()
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)

    vector_db = Qdrant(client=qdrant_client, collection_name="GPT4ALL", embeddings=embedding)
    logger.info("SUCCESS: Chroma DB initialized.")

    return vector_db


def embedd_documents_gpt4all(dir: str) -> None:
    """embedd_documents embedds the documents in the given directory.

    :param cfg: Configuration from the file
    :type cfg: DictConfig
    :param dir: PDF Directory
    :type dir: str
    """
    vector_db: Qdrant = get_db_connection()

    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    logger.info(f"Loaded {len(docs)} documents.")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vector_db.add_texts(texts=texts, metadatas=metadatas)
    logger.info("SUCCESS: Texts embedded.")


def summarize_text_gpt4all(text: str) -> str:
    """Summarize text with GPT4ALL."""
    prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text, language="de")

    model = GPT4All("orca-mini-3b.ggmlv3.q4_0.bin")

    output = model.generate(prompt, max_tokens=300)

    return output


def complete_text_gpt4all(text: str, query: str) -> str:
    """Complete text with GPT4ALL."""
    prompt = generate_prompt(prompt_name="gpt4all-completion.j2", text=text, query=query, language="de")

    model = GPT4All("orca-mini-3b.ggmlv3.q4_0.bin")

    output = model.generate(prompt, max_tokens=300)

    return output


def search_documents_openai(query: str, amount: int) -> List[Tuple[Document, float]]:
    """Searches the documents in the Chroma DB with a specific query.

    Args:
        open_ai_token (str): The OpenAI API token.
        query (str): The question for which documents should be searched.

    Returns:
        List[Tuple[Document, float]]: A list of search results, where each result is a tuple
        containing a Document object and a float score.
    """
    vector_db = get_db_connection()

    docs = vector_db.similarity_search_with_score(query, k=amount)
    logger.info("SUCCESS: Documents found.")
    return docs


def completion_gpt4all():
    """Complete text with GPT4ALL."""
    pass
