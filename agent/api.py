"""FastAPI Backend for the Knowledge Agent."""
import os
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.openapi.utils import get_openapi
from langchain.docstore.document import Document as LangchainDocument
from loguru import logger
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from agent.backend.aleph_alpha_service import (
    embedd_documents_aleph_alpha,
    embedd_text_aleph_alpha,
    embedd_text_files_aleph_alpha,
    explain_completion,
    process_documents_aleph_alpha,
    qa_aleph_alpha,
    search_documents_aleph_alpha,
    summarize_text_aleph_alpha,
)
from agent.backend.gpt4all_service import (
    embedd_documents_gpt4all,
    search_documents_gpt4all,
)
from agent.backend.open_ai_service import (
    embedd_documents_openai,
    search_documents_openai,
)
from agent.utils.utility import combine_text_from_list

# add file logger for loguru
logger.add("logs/file_{time}.log", backtrace=True, diagnose=True)
logger.info("Startup.")


def my_schema() -> dict:
    """Used to generate the OpenAPI schema.

    Returns:
        _type_: _description_
    """
    openapi_schema = get_openapi(
        title="Conversational AI API",
        version="1.0",
        description="Chat with your Documents using Conversational AI by Aleph Alpha and OpenAI.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# initialize the Fast API Application.
app = FastAPI(debug=True)
app.openapi = my_schema

load_dotenv()


# load the token from the environment variables, is None if not set.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALEPH_ALPHA_API_KEY = os.environ.get("ALEPH_ALPHA_API_KEY")
logger.info("Loading REST API Finished.")


class QARequest(BaseModel):
    """Request for the QA endpoint."""

    query: Optional[str] = Field(None, title="Query", description="The question to answer.")
    llm_backend: str = Field("openai", title="LLM Provider", description="The LLM provider to use for answering the question. Can be 'openai' or 'aleph-alpha'.")
    token: Optional[str] = Field(None, title="API Token", description="The API token for the LLM provider.")
    amount: int = Field(1, title="Amount", description="The number of answers to return.")
    language: str = Field("de", title="Language", description="The language to use for the answer.")
    history: int = Field(0, title="History", description="The number of previous questions to include in the context.")
    history_list: List[str] = Field(None, title="History List", description="A list of previous questions to include in the context.")


class EmbeddTextFilesRequest(BaseModel):
    """The request for the Embedd Text Files endpoint."""

    files: List[UploadFile] = Field(..., description="The list of text files to embed.")
    llm_backend: str = Field("openai", description="The LLM provider to use for embedding.")
    token: Optional[str] = Field(None, description="The API token for the LLM provider.")
    seperator: str = Field("###", description="The seperator to use between embedded texts.")


class SearchRequest(BaseModel):
    """The request parameters for searching the database."""

    query: str = Field(..., title="Query", description="The search query.")
    llm_backend: str = Field("openai", title="LLM Provider", description="The LLM provider to use for searching.")
    token: Optional[str] = Field(None, title="API Token", description="The API token for the LLM provider.")
    amount: int = Field(3, title="Amount", description="The number of search results to return.")


class EmbeddTextRequest(BaseModel):
    """The request parameters for embedding text."""

    text: str = Field(..., title="Text", description="The text to embed.")
    file_name: str = Field(..., title="File Name", description="The name of the file to save the embedded text to.")
    llm_backend: str = Field("openai", title="LLM Provider", description="The LLM provider to use for embedding.")
    token: Optional[str] = Field(None, title="API Token", description="The API token for the LLM provider.")
    seperator: str = Field("###", title="seperator", description="The seperator to use between embedded texts.")


class ExplainRequest(BaseModel):
    """The request parameters for explaining the output."""

    prompt: str = Field(..., title="Prompt", description="The prompt used to generate the output.")
    output: str = Field(..., title="Output", description="The output to be explained.")
    token: Optional[str] = Field(None, title="API Token", description="The Aleph Alpha API token.")


def get_token(token: Optional[str], llm_backend: str) -> str:
    """Get the token from the environment variables or the parameter.

    Args:
        token (str, optional): Token from the REST service.
        llm_backend (str): LLM provider. Defaults to "openai".

    Returns:
        str: Token for the LLM Provider of choice.

    Raises:
        ValueError: If no token is provided.
    """
    env_token = ALEPH_ALPHA_API_KEY if llm_backend in {"aleph-alpha", "aleph_alpha", "aa"} else OPENAI_API_KEY
    if env_token is None and token is None:
        raise ValueError("No token provided.")
    return token or env_token  # type: ignore


@app.get("/")
def read_root() -> str:
    """Returns the welcome message.

    Returns:
        str: The welcome message.
    """
    return "Welcome to the Simple Aleph Alpha FastAPI Backend!"


def embedd_documents_wrapper(folder_name: str, llm_backend: str = "openai", token: Optional[str] = None) -> None:
    """Call the right embedding function for the chosen backend.

    Args:
        folder_name (str): Name of the temporary folder.
        llm_backend (str, optional): LLM provider. Defaults to "openai".
        token (str, optional): Token for the LLM Provider of choice. Defaults to None.

    Raises:
        ValueError: If an invalid LLM Provider is set.
    """
    token = get_token(token, llm_backend)
    if token is None:
        raise ValueError("Please provide a token for the LLM Provider of choice.")

    if llm_backend in {"aleph-alpha", "aleph_alpha", "aa"}:
        # Embedd the documents with Aleph Alpha
        embedd_documents_aleph_alpha(dir=folder_name, aleph_alpha_token=token)
    elif llm_backend == "openai":
        # Embedd the documents with OpenAI
        embedd_documents_openai(dir=folder_name, open_ai_token=token)
    elif llm_backend == "gpt4all":
        embedd_documents_gpt4all(dir=folder_name)
    else:
        raise ValueError("Please provide either 'aleph-alpha' or 'openai' as a parameter. Other backends are not implemented yet.")


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


@app.post("/embedd_documents")
async def upload_documents(files: List[UploadFile] = File(...), llm_backend: str = "openai", token: Optional[str] = None) -> JSONResponse:
    """Uploads multiple documents to the backend.

    Args:
        files (List[UploadFile], optional): Uploaded files. Defaults to File(...).

    Returns:
        JSONResponse: The response as JSON.
    """
    token = get_token(token, llm_backend)
    if not token:
        raise ValueError("Please provide a token for the LLM Provider of choice.")
    tmp_dir = create_tmp_folder()

    file_names = []

    for file in files:
        file_name = file.filename
        file_names.append(file_name)

        # Save the file to the temporary folder
        if tmp_dir is None or not os.path.exists(tmp_dir):
            raise ValueError("Please provide a temporary folder to save the files.")

        if file_name is None:
            raise ValueError("Please provide a file to save.")

        with open(os.path.join(tmp_dir, file_name), "wb") as f:
            f.write(await file.read())

    embedd_documents_wrapper(folder_name=tmp_dir, llm_backend=llm_backend, token=token)
    return JSONResponse(content={"message": "Files received and saved.", "filenames": file_names})


@app.post("/embedd_document/")
async def embedd_one_document(file: UploadFile, llm_backend: str = "openai", token: Optional[str] = None) -> JSONResponse:
    """Uploads one document to the backend and embeds it in the database.

    Args:
        file (UploadFile): The file to upload. Should be a PDF file.
        llm_backend (str, optional): The backend to use. Defaults to "openai".
        token (str, optional): The API token. Defaults to None.

    Raises:
        ValueError: If the backend is not implemented yet.

    Returns:
        JSONResponse: A response indicating which files were received and saved.
    """
    token = get_token(token, llm_backend)
    if not token:
        raise ValueError("Please provide a token for the LLM Provider of choice.")
    # Create a temporary folder to save the files
    tmp_dir = create_tmp_folder()

    tmp_file_path = os.path.join(tmp_dir, str(file.filename))

    logger.info(tmp_file_path)
    print(tmp_file_path)

    with open(tmp_file_path, "wb") as f:
        f.write(await file.read())

    embedd_documents_wrapper(folder_name=tmp_dir, llm_backend=llm_backend, token=token)
    return JSONResponse(content={"message": "File received and saved.", "filenames": file.filename})


@app.post("/embedd_text/")
async def embedd_text(request: EmbeddTextRequest) -> JSONResponse:
    """Embeds text in the database.

    Args:
        request (EmbeddTextRequest): The request parameters.

    Raises:
        ValueError: If no token is provided or if no LLM provider is specified.

    Returns:
        JSONResponse: A response indicating that the text was received and saved, along with the name of the file it was saved to.
    """
    token = get_token(request.token, request.llm_backend)
    if token is None:
        raise ValueError("Please provide a token for the LLM Provider of choice.")

    if request.llm_backend in {"aleph-alpha", "aleph_alpha", "aa"}:
        # Embedd the documents with Aleph Alpha
        embedd_text_aleph_alpha(text=request.text, file_name=request.file_name, aleph_alpha_token=token, seperator=request.seperator)
    elif request.llm_backend == "openai":
        # Embedd the documents with OpenAI
        raise ValueError("Not implemented yet.")
    elif request.llm_backend == "gpt4all":
        # embedd_documents_gpt4all(dir=)
        raise ValueError("Not implemented yet.")
    else:
        raise ValueError("Please provide either 'aleph-alpha', 'gpt4all' or 'openai' as a parameter. Other backends are not implemented yet.")


@app.post("/embedd_text_file/")
async def embedd_text_files(request: EmbeddTextFilesRequest) -> JSONResponse:
    """Embeds text files in the database.

    Args:
        request (EmbeddTextFilesRequest): The request parameters.

    Raises:
        ValueError: If a file does not have a valid name, if no temporary folder is provided, or if no token or LLM provider is specified.

    Returns:
        JSONResponse: A response indicating that the files were received and saved, along with the names of the files they were saved to.
    """
    tmp_dir = create_tmp_folder()

    file_names = []

    for file in request.files:
        file_name = file.filename
        file_names.append(file_name)

        if file_name is None:
            raise ValueError("File does not have a valid name.")

        # Save the files to the temporary folder
        if tmp_dir is None or not os.path.exists(tmp_dir):
            raise ValueError("Please provide a temporary folder to save the files.")

        with open(os.path.join(tmp_dir, file_name), "wb") as f:
            f.write(await file.read())

    token = get_token(request.token, request.llm_backend)
    if token is None:
        raise ValueError("Please provide a token for the LLM Provider of choice.")

    if request.llm_backend is None:
        raise ValueError("Please provide a LLM Provider of choice.")

    embedd_text_files_aleph_alpha(folder=tmp_dir, aleph_alpha_token=token, seperator=request.seperator)

    return JSONResponse(content={"message": "Files received and saved.", "filenames": file_names})


@app.post("/search")
def search(request: SearchRequest) -> JSONResponse:
    """Searches for a query in the vector database.

    Args:
        request (SearchRequest): The search request.

    Raises:
        ValueError: If the LLM provider is not implemented yet.

    Returns:
        List[str]: A list of matching documents.
    """
    token = get_token(request.token, request.llm_backend)
    if token is None:
        raise ValueError("Please provide a token for the LLM Provider of choice.")

    if request.llm_backend is None:
        raise ValueError("Please provide a LLM Provider of choice.")

    DOCS = search_database(query=request.query, llm_backend=request.llm_backend, token=token, amount=request.amount)

    return JSONResponse(content={"documents": DOCS})


@app.post("/qa")
def question_answer(request: QARequest) -> JSONResponse:
    """Answer a question based on the documents in the database.

    Args:
        request (QARequest): The request parameters.

    Raises:
        ValueError: Error if no query or token is provided.

    Returns:
        Tuple: Answer, Prompt and Meta Data
    """
    # if the query is not provided, raise an error
    if request.query is None:
        raise ValueError("Please provide a Question.")

    token = get_token(request.token, request.llm_backend)
    if not token:
        raise ValueError("Please provide a token.")

    # if the history flag is activated and the history is not provided, raise an error
    if request.history and request.history is None:
        raise ValueError("Please provide a HistoryList.")

    # summarize the history
    if request.history:
        # combine the texts
        text = combine_text_from_list(request.history_list)

        if request.llm_backend in {"aleph-alpha", "aleph_alpha", "aa"}:

            # summarize the text
            summary = summarize_text_aleph_alpha(text=text, token=token)
            # combine the history and the query
            request.query = f"{summary}\n{request.query}"

        elif request.llm_backend == "openai":
            pass

        else:
            raise ValueError("Please provide either 'aleph-alpha' or 'openai' as a parameter. Other backends are not implemented yet.")

    documents = search_database(query=request.query, llm_backend=request.llm_backend, token=token, amount=request.amount)

    # call the qa function
    answer, prompt, meta_data = qa_aleph_alpha(query=request.query, documents=documents, aleph_alpha_token=token)

    return JSONResponse(content={"answer": answer, "prompt": prompt, "meta_data": meta_data})


@app.post("/explain")
def explain_output(request: ExplainRequest) -> JSONResponse:
    """Explain the output of the question answering system.

    Args:
        request (ExplainRequest): The explain request.

    Raises:
        ValueError: If no token is provided.

    Returns:
        Dict[str, float]: A dictionary containing the prompt and the score of the output.
    """
    # explain the output
    logger.error(f"Output {request.output}")
    logger.error(f"Prompt {request.prompt}")

    # fail if prompt or output are not provided
    if request.prompt is None or request.output is None:
        raise ValueError("Please provide a prompt and output.")

    # fail if prompt or output are empty
    if request.prompt == "" or request.output == "":
        raise ValueError("Please provide a prompt and output.")

    token = get_token(request.token, llm_backend="aa")
    if token:
        explanations = explain_completion(prompt=request.prompt, output=request.output, token=token)
        return JSONResponse(content={"explanations": explanations})
    else:
        raise ValueError("Please provide a token.")


@app.post("/process_document")
async def process_document(files: List[UploadFile] = File(...), llm_backend: str = "openai", token: Optional[str] = None, type: str = "invoice") -> JSONResponse:
    """Process a document.

    Args:
        files (UploadFile): _description_
        llm_backend (str, optional): _description_. Defaults to "openai".
        token (Optional[str], optional): _description_. Defaults to None.
        type (str, optional): _description_. Defaults to "invoice".

    Returns:
        JSONResponse: _description_
    """
    token = get_token(token, llm_backend)
    if not token:
        raise ValueError("Please provide a token.")

    # Create a temporary folder to save the files
    tmp_dir = create_tmp_folder()

    file_names = []

    for file in files:
        file_name = file.filename
        file_names.append(file_name)

        # Save the file to the temporary folder
        if tmp_dir is None or not os.path.exists(tmp_dir):
            raise ValueError("Please provide a temporary folder to save the files.")

        if file_name is None:
            raise ValueError("Please provide a file to save.")

        with open(os.path.join(tmp_dir, file_name), "wb") as f:
            f.write(await file.read())

    process_documents_aleph_alpha(folder=tmp_dir, token=token, type=type)


def search_database(query: str, llm_backend: str = "openai", token: Optional[str] = None, amount: int = 3) -> List[tuple[LangchainDocument, float]]:
    """Searches the database for a query.

    Args:
        query (str): The search query.
        llm_backend (str, optional): The LLM provider. Defaults to "openai".
        token (str, optional): The API token. Defaults to None.
        amount (int, optional): The amount of search results. Defaults to 3.

    Raises:
        ValueError: If the LLM provider is not implemented yet.

    Returns:
        List: A list of documents that match the query.
    """
    token = get_token(token, llm_backend)
    if not token:
        raise ValueError("Please provide a token.")
    if llm_backend in {"aleph-alpha", "aleph_alpha", "aa"}:
        # Embedd the documents with Aleph Alpha
        documents = search_documents_aleph_alpha(aleph_alpha_token=token, query=query, amount=amount)
    elif llm_backend == "openai":
        documents = search_documents_openai(open_ai_token=token, query=query, amount=amount)
    elif llm_backend == "gpt4all":
        documents = search_documents_gpt4all(query=query, amount=amount)

        # Embedd the documents with OpenAI#
    else:
        raise ValueError("Please provide either 'aleph-alpha' or 'openai' as a parameter. Other backends are not implemented yet.")

    logger.info(f"Found {len(documents)} documents.")
    return documents
