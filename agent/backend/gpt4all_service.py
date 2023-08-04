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


qdrant_client = QdrantClient("http://localhost", port=6333, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=False)
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
    """Summarize text with GPT4ALL.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text, language="de")

    model = GPT4All("orca-mini-3b.ggmlv3.q4_0.bin")

    output = model.generate(prompt, max_tokens=300)

    return output


def completion_text_gpt4all(text: str, query: str) -> str:
    """Complete text with GPT4ALL.

    Args:
        text (str): The text as basic input.
        query (str): The query to be inserted into the template.

    Returns:
        str: The completed text.
    """
    prompt = generate_prompt(prompt_name="gpt4all-completion.j2", text=text, query=query, language="de")

    model = GPT4All("orca-mini-3b.ggmlv3.q4_0.bin")

    output = model.generate(prompt, max_tokens=300)

    return output


def search_documents_gpt4all(query: str, amount: int) -> List[Tuple[Document, float]]:
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


# def qa_gpt4all(documents: list[tuple[LangchainDocument, float]], query: str, summarization: bool = False)
# -> Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]:
#     """QA takes a list of documents and returns a list of answers.

#     Args:
#         aleph_alpha_token (str): The Aleph Alpha API token.
#         documents (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
#         query (str): The query to ask.
#         summarization (bool, optional): Whether to use summarization. Defaults to False.

#     Returns:
#         Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]: A tuple containing the answer, the prompt, and the metadata for the documents.
#     """
#     # if the list of documents contains only one document extract the text directly
#     if len(documents) == 1:
#         text = documents[0][0].page_content
#         meta_data = documents[0][0].metadata

#     else:
#         # extract the text from the documents
#         texts = [doc[0].page_content for doc in documents]
#         if summarization:
#             # call summarization
#             text = ""
#             for t in texts:
#                 text += summarize_text_gpt4all(t)

#         else:
#             # combine the texts to one text
#             text = " ".join(texts)
#         meta_data = [doc[0].metadata for doc in documents]

#     # load the prompt
#     prompt = generate_prompt("qa.j2", text=text, query=query)

#     try:

#         # call the luminous api
#         answer = completion_text_gpt4all(prompt)

#     except ValueError as e:
#         # if the code is PROMPT_TOO_LONG, split it into chunks
#         if e.args[0] == "PROMPT_TOO_LONG":
#             logger.info("Prompt too long. Summarizing.")

#             # summarize the text
#             short_text = summarize_text_gpt4all(text)

#             # generate the prompt
#             prompt = generate_prompt("qa.j2", text=short_text, query=query)

#             # call the luminous api
#             answer = completion_text_gpt4all(prompt)

#     # extract the answer
#     return answer, prompt, meta_data


if __name__ == "__main__":
    embedd_documents_gpt4all(dir="data")

    print(f'Summary: {summarize_text_gpt4all(text="Das ist ein Test.")}')

    print(f'Completion: {completion_text_gpt4all(text="Das ist ein Test.", query="Was ist das?")}')
