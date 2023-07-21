"""FastAPI Backend for the Knowledge Agent."""
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.openapi.utils import get_openapi
from langchain.docstore.document import Document as LangchainDocument
from loguru import logger
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
    summarize_text_gpt4all,
)
from agent.backend.open_ai_service import (
    embedd_documents_openai,
    search_documents_openai,
)
from agent.data_model.rest_data_model import (
    EmbeddTextFilesRequest,
    EmbeddTextRequest,
    ExplainRequest,
    QARequest,
    SearchRequest,
)
from agent.utils.utility import (
    combine_text_from_list,
    create_tmp_folder,
    validate_token,
)

# add file logger for loguru
logger.add("logs/file_{time}.log", backtrace=False, diagnose=False)
logger.info("Startup.")

# TODO: Refactor


def my_schema() -> dict:
    """Used to generate the OpenAPI schema.

    Returns:
        FastAPI: FastAPI App
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
    token = validate_token(token=token, llm_backend=llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

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


@app.post("/embedd_documents")
async def post_upload_documents(files: List[UploadFile] = File(...), llm_backend: str = "openai", token: Optional[str] = None) -> JSONResponse:
    """Uploads multiple documents to the backend.

    Args:
        files (List[UploadFile], optional): Uploaded files. Defaults to File(...).

    Returns:
        JSONResponse: The response as JSON.
    """
    token = validate_token(token=token, llm_backend=llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)
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
async def post_embedd_document(file: UploadFile, llm_backend: str = "openai", token: Optional[str] = None) -> JSONResponse:
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
    token = validate_token(token=token, llm_backend=llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)
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
    token = validate_token(token=request.token, llm_backend=request.llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    if request.llm_backend in {"aleph-alpha", "aleph_alpha", "aa"}:
        # Embedd the documents with Aleph Alpha
        embedd_text_aleph_alpha(text=request.text, file_name=request.file_name, aleph_alpha_token=token, seperator=request.seperator)
        # return a success notificaton
        return JSONResponse("Embedding Sucessful.")
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

    token = validate_token(token=request.token, llm_backend=request.llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

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
    token = validate_token(token=request.token, llm_backend=request.llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    if request.llm_backend is None:
        raise ValueError("Please provide a LLM Provider of choice.")

    DOCS = search_database(query=request.query, llm_backend=request.llm_backend, token=token, amount=request.amount)

    return JSONResponse(content={"documents": DOCS})


# TODO: REFACTOR
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

    token = validate_token(token=request.token, llm_backend=request.llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    # if the history flag is activated and the history is not provided, raise an error
    if request.history and request.history is None:
        raise ValueError("Please provide a HistoryList.")

    # summarize the history
    if request.history:
        # combine the texts
        text = combine_text_from_list(request.history_list)
        # TODO: refactor for match.
        if request.llm_backend in {"aleph-alpha", "aleph_alpha", "aa"}:

            # summarize the text
            summary = summarize_text_aleph_alpha(text=text, token=token)
            # combine the history and the query
            request.query = f"{summary}\n{request.query}"

        elif request.llm_backend == "openai":
            pass

        elif request.llm_backend == "gpt4all":
            # summarize the text
            summary = summarize_text_gpt4all(text=text)
            # combine the history and the query
            request.query = f"{summary}\n{request.query}"
        else:
            raise ValueError("Please provide either 'aleph-alpha', 'gpt4all' or 'openai' as a parameter. Other backends are not implemented yet.")

    documents = search_database(query=request.query, llm_backend=request.llm_backend, token=token, amount=request.amount)

    # call the qa function
    if request.llm_backend in {"aleph-alpha", "aleph_alpha", "aa"}:
        answer, prompt, meta_data = qa_aleph_alpha(query=request.query, documents=documents, aleph_alpha_token=token)
    elif request.llm_backend == "openai":
        # todo:
        raise ValueError("Please provide either 'aleph-alpha', 'gpt4all' or 'openai' as a parameter. Other backends are not implemented yet.")
    elif request.llm_backend == "gpt4all":
        # todo:
        raise ValueError("Please provide either 'aleph-alpha', 'gpt4all' or 'openai' as a parameter. Other backends are not implemented yet.")
    else:
        raise ValueError("Please provide either 'aleph-alpha', 'gpt4all' or 'openai' as a parameter. Other backends are not implemented yet.")

    return JSONResponse(content={"answer": answer, "prompt": prompt, "meta_data": meta_data})


@app.get("/explain-qa")
def explain_question_answer(query: Optional[str] = None, llm_backend: str = "openai", token: Optional[str] = None, amount: int = 1) -> JSONResponse:
    """Answer a question & explains it based on the documents in the database.

    This uses the normal qa but combines it with the explain function.
    Args:
        query (str, optional): _description_. Defaults to None.
        aa_or_openai (str, optional): _description_. Defaults to "openai".
        token (str, optional): _description_. Defaults to None.
        amount (int, optional): _description_. Defaults to 1.

    Raises:
        ValueError: Error if no query or token is provided.

    Returns:
        Tuple: Answer, Prompt and Meta Data
    """
    # if the query is not provided, raise an error
    if query is None:
        raise ValueError("Please provide a Question.")

    token = validate_token(token=token, llm_backend=llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    documents = search_database(query=query, llm_backend=llm_backend, token=token, amount=amount)

    # call the qa function
    answer, prompt, meta_data = qa_aleph_alpha(query=query, documents=documents, aleph_alpha_token=token)

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
    # fail if prompt or output are not provided
    if request.prompt is None or request.output is None:
        raise ValueError("Please provide a prompt and output.")

    # fail if prompt or output are empty
    if request.prompt == "" or request.output == "":
        raise ValueError("Please provide a prompt and output.")

    token = validate_token(token=request.token, llm_backend=request.llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)
    explanations = explain_completion(prompt=request.prompt, output=request.output, token=token)
    return JSONResponse(content={"explanations": explanations})


@app.post("/process_document")
async def process_document(files: List[UploadFile] = File(...), llm_backend: str = "openai", token: Optional[str] = None, type: str = "invoice") -> None:
    """Process a document.

    Args:
        files (UploadFile): _description_
        llm_backend (str, optional): _description_. Defaults to "openai".
        token (Optional[str], optional): _description_. Defaults to None.
        type (str, optional): _description_. Defaults to "invoice".

    Returns:
        JSONResponse: _description_
    """
    token = validate_token(token=token, llm_backend=llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

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
    token = validate_token(token=token, llm_backend=llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

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
