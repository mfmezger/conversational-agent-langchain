"""This script is used to initialize the chroma db backend."""
import os

from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from loguru import logger
from omegaconf import DictConfig

from agent.utils.configuration import load_config

load_dotenv()


@load_config(location="config/chroma_db.yml")
def get_db_connection_(cfg: DictConfig, open_ai_token: str):

    embedding = OpenAIEmbeddings(chunk_size=1, openai_api_key=open_ai_token)
    vector_db = Chroma(persist_directory=cfg.chroma.persist_directory, embedding_function=embedding)

    return vector_db


@load_config(location="config/chroma_db.yml")
def embedd_documents(cfg: DictConfig, dir: str, open_ai_token: str):
    vector_db = get_db_connection_(open_ai_token=open_ai_token)

    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    logger.info(f"Loaded {len(docs)} documents.")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vector_db.add_texts(texts=texts, metadatas=metadatas)

    vector_db.persist()


def search_documents(open_ai_token: str, query: str):
    vector_db = get_db_connection_(open_ai_token=open_ai_token)

    docs = vector_db.similarity_search_with_score(query, k=3)

    return docs


if __name__ == "__main__":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = "https://openaiendpoint.openai.azure.com/"
    embedd_documents("data", os.getenv("OPENAI_API_KEY"))

    DOCS = search_documents(open_ai_token="", query="Muss ich mein Mietwagen volltanken?")
    print(DOCS)
