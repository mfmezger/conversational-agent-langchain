"""Utility module."""

import uuid
from collections.abc import Sequence
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from loguru import logger

from agent.data_model.internal_model import RetrievalResults

# add new languages to detect here


def combine_text_from_list(input_list: list) -> str:
    """Combines all strings in a list to one string.

    Args:
    ----
        input_list (list): List of strings

    Raises:
    ------
        TypeError: Input list must contain only strings

    Returns:
    -------
        str: Combined string

    """
    # iterate through list and combine all strings to one

    logger.info(f"List: {input_list}")

    for text in input_list:
        # verify that text is a string
        if not isinstance(text, str):
            msg = "Input list must contain only strings"
            raise TypeError(msg)

    return "\n".join(input_list)


def load_prompt_template(prompt_name: str, task: str) -> PromptTemplate:
    """Loading a task specific prompt template.

    Args:
    ----
        prompt_name (str): Name of the prompt template
        task (str): Name of the task, e.g. chat.

    Raises:
    ------
        FileNotFoundError: If the File does not exist.

    Returns:
    -------
        PromptTemplate: The loaded prompt template

    """
    try:
        with Path(Path("prompts") / task / prompt_name).open(encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError as e:
        msg = f"Prompt file '{prompt_name}' not found."
        raise FileNotFoundError(msg) from e

    return PromptTemplate.from_template(prompt_template)


def convert_qdrant_result_to_retrieval_results(docs: list) -> list[RetrievalResults]:
    """Converts the Qdrant result to a list of tuples.

    Args:
    ----
        docs (list): The Qdrant result.

    Returns:
    -------
        list: The list of tuples.

    """
    return [RetrievalResults(document=doc[0].page_content, score=doc[1], metadata=doc[0].metadata) for doc in docs]


def create_tmp_folder() -> str:
    """Creates a temporary folder for files to store.

    Returns
    -------
        str: The directory name.

    """
    # Create a temporary folder to save the files
    tmp_dir = Path.cwd() / f"tmp_{uuid.uuid4()}"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created new folder {tmp_dir}.")
    except ValueError as e:
        logger.error(f"Failed to create directory {tmp_dir}. Error: {e}")
        raise
    return str(tmp_dir)


def format_docs_for_citations(docs: Sequence[Document]) -> str:
    """Format the documents for citations.

    Args:
    ----
        docs (Sequence[Document]): Langchain documents from a vectordatabase.

    Returns:
    -------
        str: Combined documents in a format suitable for citations.

    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)
