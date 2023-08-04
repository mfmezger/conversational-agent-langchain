"""This script is used to initialize the chroma db backend with Azure OpenAI."""
import os
from typing import List, Tuple

import openai
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models

from agent.utils.configuration import load_config
from agent.utils.utility import generate_prompt

load_dotenv()


qdrant_client = QdrantClient("http://localhost", port=6333, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=False)
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
def get_db_connection(open_ai_token: str, cfg: DictConfig) -> Qdrant:
    """get_db_connection initializes the connection to the chroma db.

    :param cfg: OmegaConf configuration
    :type cfg: DictConfig
    :param open_ai_token: OpenAI API Token
    :type open_ai_token: str
    :return: Chroma DB connection
    :rtype: Chroma
    """
    embedding = OpenAIEmbeddings(chunk_size=1, openai_api_key=open_ai_token)
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)

    vector_db = Qdrant(client=qdrant_client, collection_name="OpenAI", embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB Connection.")
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
    vector_db: Qdrant = get_db_connection(open_ai_token=open_ai_token)

    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    logger.info(f"Loaded {len(docs)} documents.")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vector_db.add_texts(texts=texts, metadatas=metadatas)
    logger.info("SUCCESS: Texts embedded.")


def search_documents_openai(open_ai_token: str, query: str, amount: int) -> List[Tuple[Document, float]]:
    """Searches the documents in the Chroma DB with a specific query.

    Args:
        open_ai_token (str): The OpenAI API token.
        query (str): The question for which documents should be searched.

    Returns:
        List[Tuple[Document, float]]: A list of search results, where each result is a tuple
        containing a Document object and a float score.
    """
    vector_db = get_db_connection(open_ai_token=open_ai_token)

    docs = vector_db.similarity_search_with_score(query, k=amount)
    logger.info("SUCCESS: Documents found.")
    return docs


@load_config(location="config/ai/openai.yml")
def summarize_text_openai(text: str, token: str, cfg: DictConfig) -> str:
    """Summarizes the given text using the Luminous API.

    Args:
        text (str): The text to be summarized.
        token (str): The token for the Luminous API.

    Returns:
        str: The summary of the text.
    """
    prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text, language="de")

    openai.api_key = token
    response = openai.Completion.create(
        engine=cfg.openai.model,
        prompt=prompt,
        temperature=cfg.openai.temperature,
        max_tokens=cfg.openai.max_tokens,
        top_p=cfg.openai.top_p,
        frequency_penalty=cfg.openai.frequency_penalty,
        presence_penalty=cfg.openai.presence_penalty,
        best_of=cfg.openai.best_of,
        stop=cfg.openai.stop,
    )

    return response.choices[0].text


@load_config(location="config/ai/openai.yml")
def send_completion(text: str, query: str, token: str, cfg: DictConfig) -> str:
    """Sent completion request to OpenAI API.

    Args:
        text (str): The text on which the completion should be based.
        query (str): The query for the completion.
        token (str): The token for the OpenAI API.
        cfg (DictConfig):

    Returns:
        str: Response from the OpenAI API.
    """
    prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text, query=query, language="de")

    openai.api_key = token
    response = openai.Completion.create(
        engine=cfg.openai.model,
        prompt=prompt,
        temperature=cfg.openai.temperature,
        max_tokens=cfg.openai.max_tokens,
        top_p=cfg.openai.top_p,
        frequency_penalty=cfg.openai.frequency_penalty,
        presence_penalty=cfg.openai.presence_penalty,
        best_of=cfg.openai.best_of,
        stop=cfg.openai.stop,
    )

    return response.choices[0].text


def qa_openai(token: str, documents: list[tuple[Document, float]], query: str, summarization: bool = False) -> str:
    """QA Function for OpenAI LLMs.

    Args:
        token (str): _description_
        documents (list[tuple[Document, float]]): _description_
        query (str): _description_
        summarization (bool, optional): _description_. Defaults to False.

    Returns:
        str: _description_
    """
    # if the list of documents contains only one document extract the text directly
    if len(documents) == 1:
        text = documents[0][0].page_content
        meta_data = documents[0][0].metadata

    else:
        # extract the text from the documents
        texts = [doc[0].page_content for doc in documents]
        if summarization:
            # call summarization
            text = ""
            for t in texts:
                text += summarize_text_openai(text=t, token=token)

        else:
            # combine the texts to one text
            text = " ".join(texts)
        meta_data = [doc[0].metadata for doc in documents]

    # load the prompt
    prompt = generate_prompt("qa.j2", text=text, query=query)

    try:

        # call the luminous api
        answer = send_completion(prompt, token)

    except ValueError as e:
        # if the code is PROMPT_TOO_LONG, split it into chunks
        if e.args[0] == "PROMPT_TOO_LONG":
            logger.info("Prompt too long. Summarizing.")

            # summarize the text
            short_text = summarize_text_openai(text, token)

            # generate the prompt
            prompt = generate_prompt("openai-qa.j2", text=short_text, query=query)

            # call the luminous api
            answer = send_completion(prompt, token)

    # extract the answer
    return answer, prompt, meta_data


if __name__ == "__main__":

    token = os.getenv("OPENAI_API_KEY")

    if not token:
        raise ValueError("OPENAI_API_KEY is not set.")

    embedd_documents_openai(dir="data", open_ai_token=token)

    DOCS = search_documents_openai(open_ai_token="", query="Was ist Vanille?", amount=3)
    print(f"DOCUMENTS: {DOCS}")

    summary = summarize_text_openai(text="Below is an extract from the annual financial report of a company. ", token=token)

    print(f"SUMMARY: {summary}")
