"""This script is used to initialize the chroma db backend with Azure OpenAI."""
import os
from typing import List, Tuple

import openai
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from loguru import logger
from omegaconf import DictConfig

from agent.utils.configuration import load_config

load_dotenv()


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


def summarize_text_openai(text: str, token: str) -> str:
    """Summarizes the given text using the Luminous API.

    Args:
        text (str): The text to be summarized.
        token (str): The token for the Luminous API.

    Returns:
        str: The summary of the text.
    """
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"

    response = openai.Completion.create(
        engine="text-davinci-003", prompt=prompt, temperature=0.3, max_tokens=350, top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1, stop=None
    )

    return response.choices[0].text


if __name__ == "__main__":

    token = os.getenv("OPENAI_API_KEY")

    if not token:
        raise ValueError("OPENAI_API_KEY is not set.")

    embedd_documents_openai("data", token)

    DOCS = search_documents_openai(open_ai_token="", query="Was ist Vanille?", amount=3)
    print(DOCS)

    summary = summarize_text_openai(
        """Below is an extract from the annual financial report of a company. """,
        token,
    )

    print(summary)
