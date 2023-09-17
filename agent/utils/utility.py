"""This is the utility module."""
import os
import uuid

from jinja2 import Template
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient

from agent.utils.configuration import load_config


def combine_text_from_list(input_list: list) -> str:
    """Combines all strings in a list to one string.

    Args:
        input_list (list): List of strings

    Raises:
        TypeError: Input list must contain only strings

    Returns:
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
            raise TypeError("Input list must contain only strings")

    return combined_text


def generate_prompt(prompt_name: str, text: str, query: str = "", language: str = "de") -> str:
    """Generates a prompt for the Luminous API using a Jinja template.

    Args:
        prompt_name (str): The name of the file containing the Jinja template.
        text (str): The text to be inserted into the template.
        query (str): The query to be inserted into the template.
        language (str): The language the query should output.

    Returns:
        str: The generated prompt.

    Raises:
        FileNotFoundError: If the specified prompt file cannot be found.
    """
    try:
        match language:
            case "en":
                lang = "en"
            case "de":
                lang = "de"
            case _:
                raise ValueError("Language not supported.")
        with open(os.path.join("prompts", lang, prompt_name)) as f:
            prompt = Template(f.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file '{prompt_name}' not found.")

    # replace the value text with jinja
    # Render the template with your variable
    if query:
        prompt_text = prompt.render(text=text, query=query)
    else:
        prompt_text = prompt.render(text=text)

    return prompt_text


def create_tmp_folder() -> str:
    """Creates a temporary folder for files to store.

    Returns:
        str: The directory name.
    """
    # Create a temporary folder to save the files
    tmp_dir = f"tmp_{str(uuid.uuid4())}"
    os.makedirs(tmp_dir)
    logger.info(f"Created new folder {tmp_dir}.")
    return tmp_dir


def get_token(token: str | None, llm_backend: str | None, aleph_alpha_key: str | None, openai_key: str | None) -> str:
    """Get the token from the environment variables or the parameter.

    Args:
        token (str, optional): Token from the REST service.
        llm_backend (str): LLM provider. Defaults to "openai".

    Returns:
        str: Token for the LLM Provider of choice.

    Raises:
        ValueError: If no token is provided.
    """
    env_token = aleph_alpha_key if llm_backend in {"aleph-alpha", "aleph_alpha", "aa"} else openai_key
    if not env_token and not token:
        raise ValueError("No token provided.")  #

    return token or env_token  # type: ignore


def validate_token(token: str | None, llm_backend: str, aleph_alpha_key: str | None, openai_key: str | None) -> str:
    """Test if a token is available, and raise an error if it is missing when needed.

    Args:
        token (str): Token from the request
        llm_backend (str): Backend from the request
        aleph_alpha_key (str): Key from the .env file
        openai_key (str): Key from the .env file

    Raises:
        ValueError: If the llm backend is AA or OpenAI and there is no token.

    Returns:
        str: Token
    """
    if llm_backend != "gpt4all":
        token = get_token(token, llm_backend, aleph_alpha_key, openai_key)
    else:
        token = "gpt4all"
    return token


@load_config("config/db.yml")
def load_vec_db_conn(cfg: DictConfig) -> QdrantClient:
    """Load the Vector Database Connection."""
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)
    return qdrant_client


if __name__ == "__main__":
    # test the function
    generate_prompt("qa.j2", "This is a test text.", "What is the meaning of life?")
