"""The script to initialize the chroma db backend with aleph alpha."""
import os
from typing import List, Tuple

import numpy as np
from aleph_alpha_client import Client, CompletionRequest, ExplanationRequest, Prompt
from dotenv import load_dotenv
from jinja2 import Template
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain.vectorstores import Chroma
from loguru import logger
from omegaconf import DictConfig

from agent.utils.configuration import load_config

load_dotenv()


def generate_prompt(prompt_name: str, text: str, query: str) -> str:
    """Generate the prompt for the luminous api out of a jinja template.

    :param prompt_name: Name of the File with the jinja template
    :type prompt_name: str
    :param text: The text to be inserted into the template
    :type text: str
    :return: The generated prompt
    :rtype: str
    """
    with open(os.path.join("prompts", prompt_name)) as f:
        prompt = Template(f.read())

    # replace the value text with jinja
    # Render the template with your variable
    prompt = prompt.render(text=text, query=query)

    return prompt


def send_completion_request(text: str, token: str) -> str:
    """Send the request to the luminous api.

    :param text: The prompt to be sent to the api
    :type text: str
    :param token: The token for the luminous api
    :type token: str
    :return: The response from the api
    :rtype: str
    """
    client = Client(token=token)
    request = CompletionRequest(prompt=Prompt.from_text(text), maximum_tokens=256, stop_sequences=["###"])
    response = client.complete(request, model="luminous-supreme-control")

    return response.completions[0].completion


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


def search_documents_aleph_alpha(aleph_alpha_token: str, query: str, amount: int = 1) -> List[Tuple[Document, float]]:
    """search_documents takes a query and searchs the Chroma DB for similar documents.

    :param aleph_alpha_token: Aleph Alpha API Token
    :type aleph_alpha_token: str
    :param query: The Query that should be searched for.
    :type query: str
    :return: Multiple Documents
    :rtype: List[Tuple[Document, float]]
    """
    vector_db = get_db_connection(aleph_alpha_token=aleph_alpha_token)

    docs = vector_db.similarity_search_with_score(query, k=amount)
    logger.info("SUCCESS: Documents found.")
    return docs


def summarization(aleph_alpha_token: str, documents: List[Tuple[Document, float]]) -> List[str]:
    """Summarization takes a list of documents and returns a list of summaries.

    :param aleph_alpha_token: Aleph Alpha API Token
    :type aleph_alpha_token: str
    :param documents: List of documents
    :type documents: List[Tuple[Document, float]]
    :return: List of summaries
    :rtype: List[str]
    """
    client = Client(token=aleph_alpha_token)

    # TODO: Implement
    # extract the text from the documents
    texts = [doc[0].page_content for doc in documents]

    # summarize the texts
    summaries = []
    for text in texts:
        pass


def qa_aleph_alpha(aleph_alpha_token: str, documents: List[Tuple[Document, float]], query: str, summarization: bool = False) -> List[str]:
    """Qa takes a list of documents and returns a list of answers.

    :param aleph_alpha_token: Aleph Alpha API Token
    :type aleph_alpha_token: str
    :param documents: List of documents
    :type documents: List[Tuple[Document, float]]
    :return: List of answers
    :rtype: List[str]
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
            # TODO: implement summarization
            pass
        else:
            # combine the texts to one text
            text = " ".join(texts)
        meta_data = [doc[0].metadata for doc in documents]

    # load the prompt
    prompt = generate_prompt("qa.txt", text=text, query=query)
    # call the luminous api
    answer = send_completion_request(prompt, aleph_alpha_token)

    # extract the answer
    return answer, prompt, meta_data


def explain_completion(prompt: str, output: str, token: str):
    """Explain_completion takes a prompt and an output and returns the explanation.

    :param prompt: The complete input in the model
    :type prompt: str
    :param output: the answer of the model
    :type output: str
    :param token: Aleph Alpha API Token
    :type token: str
    :return: Key: Sentence, Value: Score
    :rtype: dict
    """
    exp_req = ExplanationRequest(Prompt.from_text(prompt), output, control_factor=0.1, prompt_granularity="sentence")
    client = Client(token=token)
    response_explain = client.explain(exp_req, model="luminous-supreme-control")

    explanations = response_explain[1][0].items[0][0]

    # remove the first explanation because it is the prompt
    # explanations = explanations[3:]

    # sort the explanations by score
    explanations = sorted(explanations, key=lambda x: x.score, reverse=True)

    result = {}
    # extract the first 3 explanations
    for item in explanations[:3]:
        start = item.start
        end = item.start + item.length
        result[prompt[start:end]] = np.round(item.score, decimals=3)

    return result


if __name__ == "__main__":
    embedd_documents_aleph_alpha("data", os.getenv("ALEPH_ALPHA_API_KEY"))

    DOCS = search_documents_aleph_alpha(aleph_alpha_token=os.getenv("ALEPH_ALPHA_API_KEY"), query="Muss ich mein Mietwagen volltanken?")
    logger.info(DOCS)
    answer, prompt, meta_data = qa_aleph_alpha(aleph_alpha_token=os.getenv("ALEPH_ALPHA_API_KEY"), documents=DOCS, query="Muss ich mein Mietwagen volltanken?")
    logger.info(f"Answer: {answer}")
    explanations = explain_completion(prompt, answer, os.getenv("ALEPH_ALPHA_API_KEY"))

    print(explanations)
