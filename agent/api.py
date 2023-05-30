"""FastAPI Backend for the Knowledge Agent."""
import os
import uuid
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from starlette.responses import JSONResponse

from agent.backend.aleph_alpha_service import (
    embedd_documents_aleph_alpha,
    explain_completion,
    qa_aleph_alpha,
    search_documents_aleph_alpha,
)
from agent.backend.open_ai_service import (
    embedd_documents_openai,
    search_documents_openai,
)

# initialize the Fast API Application.
app = FastAPI(debug=True)

# load the token from the environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALEPH_ALPHA_API_KEY = os.environ.get("ALEPH_ALPHA_API_KEY")


class Backend(enum.Enum):
    """Enum for the LLM Provider."""

    ALEPH_ALPHA = "aleph-alpha"
    OPENAI = "openai"


def get_token(token: str, aa_or_openai: str) -> str:
    """Get the token from the environment variables or the parameter.

    :param token: token from rest service
    :type token: str
    :param aa_or_openai: LLM provider, defaults to "openai"
    :type aa_or_openai: str
    :return: Token for the LLM Provider of choice
    :rtype: str
    """
    env_token = ALEPH_ALPHA_API_KEY if aa_or_openai in {"aleph-alpha", "aleph_alpha", "aa"} else OPENAI_API_KEY
    return token if env_token is None else env_token


load_dotenv()


@app.get("/")
def read_root() -> str:
    """Root Message.

    :return: Welcome Message
    :rtype: string
    """
    return "Welcome to the Simple Aleph Alpha FastAPI Backend!"


async def embed_documents_wrapper(folder_path: str, backend: str = "openai", token: str = None) -> None:
    """Call the right embedding function for the chosen backend.

    :param folder_path: The path to the folder containing the documents to embed.
    :type folder_path: str
    :param backend: The name of the embedding backend to use (either "aleph-alpha" or "openai").
    :type backend: str, optional
    :param token: The API token to use for the embedding backend.
    :type token: str, optional
    """
    if backend not in ["aleph-alpha", "openai"]:
        raise ValueError("Invalid backend name. Please choose either 'aleph-alpha' or 'openai'.")

    try:
        if backend == "aleph-alpha":
            # Embed the documents with Aleph Alpha
            embed_documents_aleph_alpha(folder_path, token)
        elif backend == "openai":
            # Embed the documents with OpenAI
            embed_documents_openai(folder_path, token)
    except Exception as e:
        logging.error(f"Error embedding documents: {e}")


def create_tmp_folder() -> str:
    """Creates a temporary folder for files to store.

    :return: The directory name
    :rtype: str
    """
    # Create a temporary folder to save the files
    tmp_dir = Path(f"tmp_{str(uuid.uuid4())}")
    tmp_dir.mkdir()
    logging.info(f"Created new folder {tmp_dir}.")
    return str(tmp_dir)


@app.post("/embedd_documents")
async def upload_documents(files: List[UploadFile] = File(...), aa_or_openai: str = "openai", token: str = None) -> JSONResponse:
    """Upload multiple documents to the backend.

    :param files: Uploaded files, defaults to File(...)
    :type files: List[UploadFile], optional
    :return: Return as JSON
    :rtype: JSONResponse
    """
    tmp_dir = await create_tmp_folder()

    file_names = [file.filename for file in files]

    for file in files:
        file_path = Path(tmp_dir) / file.filename
        async with file_path.open("wb") as f:
            f.write(await file.read())

    embed_documents_wrapper(folder_path=tmp_dir, backend=aa_or_openai, token=token)
    return JSONResponse(content={"message": "Files received and saved.", "filenames": file_names})


@app.post("/embed_documents")
async def embed_one_document(file: UploadFile, backend: str = "openai", token: Optional[str] = None) -> JSONResponse:
    """Upload one document to the backend.

    To embed the document in the database it is necessary to provide the name of the backend
    as well as the fitting token for that backend.

    :param file: File that is uploaded, should be a pdf file.
    :type file: UploadFile
    :param backend: Backend to use, defaults to "openai"
    :type backend: str, optional
    :param token: Token for the backend, defaults to None
    :type token: str, optional
    :return: Response which Files were received and saved.
    :rtype: JSONResponse
    """
    # Create a temporary folder to save the files
    tmp_folder_path = await create_tmp_folder()

    tmp_file_path = Path(tmp_folder_path) / file.filename

    logging.info(tmp_file_path)

    async with tmp_file_path.open("wb") as f:
        f.write(await file.read())

    embed_documents_wrapper(folder_path=tmp_folder_path, backend=backend, token=token)
    return JSONResponse(content={"message": "File received and saved.", "filenames": file.filename})


@app.get("/search")
def search(query: str, aa_or_openai: str = "openai", token: str = None, amount: int = 3) -> None:
    """Search for a query in the vector database.

    :param query: The search query
    :type query: str
    :param aa_or_openai: The LLM Provider, defaults to "openai"
    :type aa_or_openai: str, optional
    :param token: Token for the LLM Provider, defaults to None
    :type token: str, optional
    :raises ValueError: If the LLM Provider is not implemented yet
    """
    token = get_token(token, aa_or_openai)
    return search_db(query=query, aa_or_openai=aa_or_openai, token=token, amount=amount)


@app.get("/qa")
def question_answer(query: str = None, aa_or_openai: str = "openai", token: str = None, amount: int = 1):
    """Answer a question based on the documents in the database.

    :param query: _description_
    :type query: str
    :param aa_or_openai: _description_, defaults to "openai"
    :type aa_or_openai: str, optional
    :param token: _description_, defaults to None
    :type token: str, optional
    :param amount: _description_, defaults to 1
    :type amount: int, optional
    """
    # if the query is not provided, raise an error
    if query is None:
        raise ValueError("Please provide a Question.")

    token = get_token(token, aa_or_openai)
    documents = search_db(query=query, aa_or_openai=aa_or_openai, token=token, amount=amount)

    # call the qa function
    answer, prompt, meta_data = qa_aleph_alpha(query=query, documents=documents, aleph_alpha_token=token)

    return answer, prompt, meta_data


@app.post("/explain")
def explain_output(prompt: str, output: str, token: str = None):
    """Explain the output of the question answering system.

    :param prompt: _description_
    :type prompt: str
    :param answer: _description_
    :type answer: str
    :param token: _description_, defaults to None
    :type token: str, optional
    """
    # explain the output
    logger.info(f"OUtput {output}")
    token = get_token(token, aa_or_openai="aa")
    return explain_completion(prompt=prompt, output=output, token=token)


def search_db(query: str, backend: Backend = Backend.OPENAI, token: str = None, amount: int = 3) -> List:
    """Search the database for a query.

    :param query: Search query
    :type query: str
    :param backend: LLM Provider, defaults to Backend.OPENAI
    :type backend: Backend, optional
    :param token: API Token, defaults to None
    :type token: str, optional
    :param amount: Amount of search results, defaults to 3
    :type amount: int, optional
    :raises ValueError: If the LLM Provider is not implemented yet
    :return: Documents that match the query
    :rtype: List
    """
    token = get_token(token, backend.value)

    if backend == Backend.ALEPH_ALPHA:
        # Embed the documents with Aleph Alpha
        documents = search_documents_aleph_alpha(aleph_alpha_token=token, query=query, amount=amount)
    elif backend == Backend.OPENAI:
        documents = search_documents_openai(open_ai_token=token, query=query, amount=amount)
        # Embed the documents with OpenAI
    else:
        raise ValueError(f"Backend {backend} is not implemented yet.")

    logger.info(f"Found {len(documents)} documents.")

    return documents
