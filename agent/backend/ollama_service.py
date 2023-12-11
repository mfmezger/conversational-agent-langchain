"""ollama Backend Service."""
import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient

from agent.utils.configuration import load_config
from agent.utils.utility import generate_prompt

load_dotenv()


@load_config(location="config/db.yml")
def get_db_connection(cfg: DictConfig, collection_name: str) -> Qdrant:
    """Initializes a connection to the Qdrant DB.

    Args:
        cfg (DictConfig): The configuration file loaded via OmegaConf.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
        Qdrant: The Qdrant DB connection.
    """
    embedding = OllamaEmbeddings(base_url="http://host.docker.internal:11434", model="zephyr")  #
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)
    if collection_name is None or collection_name == "":
        collection_name = "ollama"
    vector_db = Qdrant(client=qdrant_client, collection_name="ollama", embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db


def embedd_documents_ollama(dir: str, collection_name: Optional[str] = None) -> None:
    """embedd_documents embedds the documents in the given directory.

    :param cfg: Configuration from the file
    :type cfg: DictConfig
    :param dir: PDF Directory
    :type dir: str
    """
    vector_db: Qdrant = get_db_connection(collection_name=collection_name)

    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
    splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = loader.load_and_split(splitter)

    logger.info(f"Loaded {len(docs)} documents.")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vector_db.add_texts(texts=texts, metadatas=metadatas)
    logger.info("SUCCESS: Texts embedded.")


def embedd_text_ollama(text: str, file_name: str, seperator: str, collection_name: Optional[str] = None) -> None:
    """embedd_documents embedds the documents in the given directory.

    :param cfg: Configuration from the file
    :type cfg: DictConfig
    :param dir: PDF Directory
    :type dir: str
    """
    vector_db: Qdrant = get_db_connection(collection_name=collection_name)

    # split the text at the seperator
    text_list: List = text.split(seperator)

    # check if first and last element are empty
    if not text_list[0]:
        text_list.pop(0)
    if not text_list[-1]:
        text_list.pop(-1)

    metadata = file_name
    # add _ and an incrementing number to the metadata
    metadata_list: List = [{"source": f"{metadata}_{str(i)}", "page": 0} for i in range(len(text_list))]

    vector_db.add_texts(texts=text_list, metadatas=metadata_list)
    logger.info("SUCCESS: Text embedded.")


def summarize_text_ollama(text: str) -> str:
    """Summarize text with ollama.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text, language="de")

    model = Ollama(base_url="http://host.docker.internal:11434", model="zephyr")

    return model(prompt)


def completion_text_ollama(prompt: str) -> str:
    """Complete text with ollama.

    Args:
        text (str): The text as basic input.
        query (str): The query to be inserted into the template.

    Returns:
        str: The completed text.
    """
    model = Ollama(base_url="http://host.docker.internal:11434", model="zephyr")

    return model(prompt)


def search_documents_ollama(query: str, amount: int, threshold: float = 0.0, collection_name: Optional[str] = None) -> List[Tuple[Document, float]]:
    """Searches the documents in the Qdrant DB with a specific query.

    Args:
        open_ai_token (str): The OpenAI API token.
        query (str): The question for which documents should be searched.

    Returns:
        List[Tuple[Document, float]]: A list of search results, where each result is a tuple
        containing a Document object and a float score.
    """
    vector_db: Qdrant = get_db_connection(collection_name=collection_name)

    docs = vector_db.similarity_search_with_score(query=query, k=amount, score_threshold=threshold)
    logger.info("SUCCESS: Documents found.")
    return docs


# def qa_chain():
#     # QA chain
#     chat_model = ChatOllama(model="zephyr", verbose=True)

#     vectorstore = get_db_connection()
#     QA_CHAIN_PROMPT = PromptTemplate(
#         input_variables=["text", "question"],
#         template=,
#     )
#     qa_chain = RetrievalQA.from_chain_type(
#         chat_model,
#         retriever=vectorstore.as_retriever(),
#         chain_type_kwargs={"prompt": prompt},
#     )
#     result = qa_chain({"query": question})


def qa_ollama(documents: list[tuple[Document, float]], query: str, summarization: bool = False, language: str = "de"):
    """QA takes a list of documents and returns a list of answers.

    Args:
        aleph_alpha_token (str): The Aleph Alpha API token.
        documents (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
        query (str): The query to ask.
        summarization (bool, optional): Whether to use summarization. Defaults to False.

    Returns:
        Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]: A tuple containing the answer, the prompt, and the metadata for the documents.
    """
    # if the list of documents contains only one document extract the text directly
    if len(documents) == 1:
        text = documents[0][0].page_content
        meta_data = documents[0][0].metadata

    else:
        # extract the text from the documents
        texts = [doc[0].page_content for doc in documents]
        if summarization:
            text = "".join(summarize_text_ollama(t) for t in texts)
        else:
            # combine the texts to one text
            text = " ".join(texts)
        meta_data = [doc[0].metadata for doc in documents]

        # combine the metadata list of dicts to one metadata dicts
        meta_data = {k: v for d in meta_data for k, v in d.items()}

    # load the prompt
    prompt = generate_prompt("ollama-qa.j2", text=text, query=query, language="en")

    try:

        # call the luminous api
        logger.info("starting completion")
        answer = completion_text_ollama(prompt)
        logger.info(f"completion done with answer {answer}")

    except ValueError as e:
        # if the code is PROMPT_TOO_LONG, split it into chunks
        if e.args[0] == "PROMPT_TOO_LONG":
            logger.info("Prompt too long. Summarizing.")

            # summarize the text
            short_text = summarize_text_ollama(text)

            # generate the prompt
            prompt = generate_prompt("ollama-qa.j2", text=short_text, query=query, language=language)

            # call the luminous api
            answer = completion_text_ollama(prompt)

        logger.error(f"Error: {e}")

    logger.info(f"Answer: {answer}, Meta Data: {meta_data}, Prompt: {prompt}")
    # extract the answer
    return answer, prompt, meta_data


if __name__ == "__main__":
    # embedd_documents_ollama(dir="tests/resources/")

    # print(f'Summary: {summarize_text_ollama(text="Das ist ein Test.")}')

    # print(f'Completion: {completion_text_ollama(text="Das ist ein Test.", query="Was ist das?")}')

    answer, prompt, meta_data = qa_ollama(documents=search_documents_ollama(query="Das ist ein Test.", amount=3), query="Was ist das?")

    logger.info(f"Answer: {answer}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Metadata: {meta_data}")
