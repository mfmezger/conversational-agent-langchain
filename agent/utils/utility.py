"""Utility module."""
import uuid
from collections.abc import Sequence
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from lingua import Language, LanguageDetectorBuilder
from loguru import logger

from agent.data_model.internal_model import RetrievalResults

# add new languages to detect here
languages = [Language.ENGLISH, Language.GERMAN]
detector = LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.7).build()


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
    combined_text = ""

    logger.info(f"List: {input_list}")

    for text in input_list:
        # verify that text is a string
        if isinstance(text, str):
            # combine the text in a new line
            combined_text += "\n".join(text)

        else:
            msg = "Input list must contain only strings"
            raise TypeError(msg)

    return combined_text


def detect_language(text: str) -> str:
    """Detect the language.

    Args:
    ----
        text (str): The input text.

    Returns:
    -------
        str: The language which was detected.

    """
    detected_lang = detector.detect_language_of(text)
    if detected_lang == "Language.ENGLISH":
        language = "en"
    elif detected_lang == "Language.GERMAN":
        language = "de"
    else:
        logger.info(f"Detected Language is not supported. Using English. Detected language was {detected_lang}.")
        language = "en"
    return language


def generate_prompt(prompt_name: str, text: str, query: str = "", language: str = "detect") -> str:
    """Generates a prompt for the Luminous API using a Jinja template.

    Args:
    ----
        prompt_name (str): The name of the file containing the Jinja template.
        text (str): The text to be inserted into the template.
        query (str): The query to be inserted into the template.
        language (str): The language the query should output. Or it can be detected

    Returns:
    -------
        str: The generated prompt.

    Raises:
    ------
        FileNotFoundError: If the specified prompt file cannot be found.

    """
    try:  # TODO: Adding the history to the prompt
        if language == "detect":
            language = detect_language(text)
        if language not in {"en", "de"}:
            msg = "Language not supported."
            raise ValueError(msg)
        with Path(Path("prompts") / language / prompt_name).open(encoding="utf-8") as f:
            prompt = PromptTemplate.from_template(f.read(), template_format="jinja2")
    except FileNotFoundError as e:
        msg = f"Prompt file '{prompt_name}' not found."
        raise FileNotFoundError(msg) from e

    return prompt.format(text=text, query=query) if query else prompt.format(text=text)


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

    return prompt_template


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


def extract_text_from_langchain_documents(docs: list[Document]) -> str:
    """Extracts the text from the langchain documents.

    Args:
    ----
        docs (list[Document]): List of Lanchain documents.

    Returns:
    -------
        str: The extracted text.
    """
    logger.info(f"Loaded {len(docs)} documents.")
    return "\n\n".join(f"Context {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))


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


if __name__ == "__main__":
    # test the function
    generate_prompt("aleph_alpha_qa.j2", "This is a test text.", "What is the meaning of life?")
