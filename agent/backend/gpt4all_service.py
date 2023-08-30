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

from agent.utils.configuration import load_config
from agent.utils.utility import generate_prompt

load_dotenv()


@load_config(location="config/db.yml")
def get_db_connection(cfg: DictConfig) -> Qdrant:
    """Initializes a connection to the Qdrant DB.

    Args:
        cfg (DictConfig): The configuration file loaded via OmegaConf.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
        Qdrant: The Qdrant DB connection.
    """
    embedding = GPT4AllEmbeddings()
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)

    vector_db = Qdrant(client=qdrant_client, collection_name="GPT4ALL", embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB initialized.")

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


def embedd_text_gpt4all(text: str, file_name: str, seperator: str) -> None:
    """embedd_documents embedds the documents in the given directory.

    :param cfg: Configuration from the file
    :type cfg: DictConfig
    :param dir: PDF Directory
    :type dir: str
    """
    vector_db: Qdrant = get_db_connection()

    # split the text at the seperator
    text_list: List = text.split(seperator)

    # check if first and last element are empty
    if not text_list[0]:
        text_list.pop(0)
    if not text_list[-1]:
        text_list.pop(-1)

    metadata = file_name
    # add _ and an incrementing number to the metadata
    metadata_list: List = [{"filename": metadata + "_" + str(i)} for i in range(len(text_list))]

    vector_db.add_texts(texts=text_list, metadatas=metadata_list)
    logger.info("SUCCESS: Text embedded.")


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


def custom_completion_prompt_gpt4all(
    prompt: str, token: str = None, model: str = "orca-mini-3b.ggmlv3.q4_0.bin", max_tokens: int = 256, stop_sequences: List[str] = ["###"], temperature: float = 0
) -> str:
    """This method sents a custom completion request to the Aleph Alpha API.

    Args:
        token (str): The token for the Aleph Alpha API.
        prompt (str): The prompt to be sent to the API.

    Raises:
        ValueError: Error if their are no completions or the completion is empty or the prompt and tokenis empty.
    """
    if not prompt:
        raise ValueError("Prompt cannot be None or empty.")

    model = GPT4All(model)

    output = model.generate(prompt, max_tokens=max_tokens, temp=temperature)

    return str(output)


def search_documents_gpt4all(query: str, amount: int) -> List[Tuple[Document, float]]:
    """Searches the documents in the Qdrant DB with a specific query.

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
