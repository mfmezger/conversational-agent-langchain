"""FastAPI Backend for the Knowledge Agent."""
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.openapi.utils import get_openapi
from langchain.docstore.document import Document as LangchainDocument
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient, models
from qdrant_client.http.models.models import UpdateResult
from starlette.responses import JSONResponse

from agent.backend.aleph_alpha_service import (
    custom_completion_prompt_aleph_alpha,
    embedd_documents_aleph_alpha,
    embedd_text_aleph_alpha,
    embedd_text_files_aleph_alpha,
    explain_completion,
    explain_qa,
    process_documents_aleph_alpha,
    qa_aleph_alpha,
    search_documents_aleph_alpha,
    summarize_text_aleph_alpha,
)
from agent.backend.gpt4all_service import (
    custom_completion_prompt_gpt4all,
    embedd_documents_gpt4all,
    embedd_text_gpt4all,
    qa_gpt4all,
    search_documents_gpt4all,
    summarize_text_gpt4all,
)
from agent.backend.open_ai_service import (
    embedd_documents_openai,
    search_documents_openai,
    send_custom_completion_openai,
)
from agent.data_model.request_data_model import (
    CustomPromptCompletion,
    EmbeddTextFilesRequest,
    EmbeddTextRequest,
    ExplainRequest,
    QARequest,
    SearchRequest,
)
from agent.data_model.response_data_model import (
    EmbeddingResponse,
    ExplainQAResponse,
    QAResponse,
    SearchResponse,
)
from agent.utils.configuration import load_config
from agent.utils.utility import (
    combine_text_from_list,
    create_tmp_folder,
    load_vec_db_conn,
    validate_token,
)

# add file logger for loguru
logger.add("logs/file_{time}.log", backtrace=False, diagnose=False)
logger.info("Startup.")


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
app.openapi = my_schema  # type: ignore

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


def embedd_documents_wrapper(folder_name: str, llm_backend: str = "aa", token: Optional[str] = None, collection_name: Optional[str] = None) -> None:
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
        logger.debug("Embedding Documents with Aleph Alpha.")
        embedd_documents_aleph_alpha(dir=folder_name, aleph_alpha_token=token, collection_name=collection_name)
    elif llm_backend == "openai":
        # Embedd the documents with OpenAI
        logger.debug("Embedding Documents with OpenAI.")
        embedd_documents_openai(dir=folder_name, open_ai_token=token)

    elif llm_backend == "gpt4all":
        embedd_documents_gpt4all(dir=folder_name)
    else:
        raise ValueError("Please provide either 'aleph-alpha' or 'openai' as a parameter. Other backends are not implemented yet.")


# generate a rest service to create a new collection
@app.post("/collection/create/{llm_provider}/{collection_name}")
def create_collection(llm_provider: str, collection_name: str) -> None:
    """Create a new collection in the vector database.

    Args:
        llm_provider (str): Name of the LLM Provider
        collection_name (str): Name of the Collection
    """
    qdrant_client, _ = initialize_qdrant_client_config()

    if llm_provider in {"aleph-alpha", "aleph_alpha", "aa"}:
        generate_collection_aleph_alpha(qdrant_client=qdrant_client, collection_name=collection_name, embeddings_size=5120)
    elif llm_provider == "OpenAI":
        generate_collection_openai(qdrant_client=qdrant_client, collection_name=collection_name)

    elif llm_provider == "gpt4all":
        generate_collection_gpt4all(qdrant_client=qdrant_client, collection_name=collection_name)


@app.post("/embeddings/documents")
async def post_embedd_documents(
    files: List[UploadFile] = File(...), llm_backend: str = "aa", token: Optional[str] = None, collection_name: Optional[str] = None
) -> EmbeddingResponse:
    """Uploads multiple documents to the backend.

    Args:
        files (List[UploadFile], optional): Uploaded files. Defaults to File(...).

    Returns:
        JSONResponse: The response as JSON.
    """
    logger.info("Embedding Multiple Documents")
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

    embedd_documents_wrapper(folder_name=tmp_dir, llm_backend=llm_backend, token=token, collection_name=collection_name)

    return EmbeddingResponse(status="success", files=file_names)


@app.post("/embeddings/document/")
async def post_embedd_document(file: UploadFile, llm_backend: str = "aa", token: Optional[str] = None, collection_name: Optional[str] = None) -> EmbeddingResponse:
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
    logger.info("Embedding Single Document")
    token = validate_token(token=token, llm_backend=llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)
    # Create a temporary folder to save the files
    tmp_dir = create_tmp_folder()

    tmp_file_path = os.path.join(tmp_dir, str(file.filename))

    logger.info(tmp_file_path)
    print(tmp_file_path)

    with open(tmp_file_path, "wb") as f:
        f.write(await file.read())

    embedd_documents_wrapper(folder_name=tmp_dir, llm_backend=llm_backend, token=token, collection_name=collection_name)
    return EmbeddingResponse(status="success", files=[file.filename])


@app.post("/embeddings/text/")
async def embedd_text(request: EmbeddTextRequest) -> EmbeddingResponse:
    """Embeds text in the database.

    Args:
        request (EmbeddTextRequest): The request parameters.

    Raises:
        ValueError: If no token is provided or if no LLM provider is specified.

    Returns:
        JSONResponse: A response indicating that the text was received and saved, along with the name of the file it was saved to.
    """
    logger.info("Embedding Text")
    token = validate_token(token=request.token, llm_backend=request.llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    logger.info(f"Requested Backend is: {request.llm_backend}")
    if request.llm_backend in {"aleph-alpha", "aleph_alpha", "aa"}:
        # Embedd the documents with Aleph Alpha
        embedd_text_aleph_alpha(text=request.text, file_name=request.file_name, aleph_alpha_token=token, seperator=request.seperator)
        # return a success notificaton
        return JSONResponse(content={"message": "Text received and saved.", "filenames": request.file_name})
    elif request.llm_backend == "openai":
        # Embedd the documents with OpenAI
        # TODO: Implement
        raise ValueError("Not implemented yet.")
    elif request.llm_backend == "gpt4all":
        embedd_text_gpt4all(text=request.text, file_name=request.file_name, seperator=request.seperator)

    else:
        raise ValueError("Please provide either 'aleph-alpha', 'gpt4all' or 'openai' as a parameter. Other backends are not implemented yet.")

    return EmbeddingResponse(status="success", files=[request.file_name])


@app.post("/embeddings/texts/files")
async def embedd_text_files(request: EmbeddTextFilesRequest) -> EmbeddingResponse:
    """Embeds text files in the database.

    Args:
        request (EmbeddTextFilesRequest): The request parameters.

    Raises:
        ValueError: If a file does not have a valid name, if no temporary folder is provided, or if no token or LLM provider is specified.

    Returns:
        JSONResponse: A response indicating that the files were received and saved, along with the names of the files they were saved to.
    """
    logger.info("Embedding Text Files")
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

    return EmbeddingResponse(status="success", files=file_names)


@app.post("/semantic/search")
def search(request: SearchRequest) -> List[SearchResponse]:
    """Searches for a query in the vector database.

    Args:
        request (SearchRequest): The search request.

    Raises:
        ValueError: If the LLM provider is not implemented yet.

    Returns:
        List[str]: A list of matching documents.
    """
    logger.info("Searching for Documents")
    token = validate_token(token=request.token, llm_backend=request.llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    if request.llm_backend is None:
        raise ValueError("Please provide a LLM Provider of choice.")

    DOCS = search_database(query=request.query, llm_backend=request.llm_backend, token=token, amount=request.amount, collection_name=request.collection_name)

    if not DOCS:
        logger.info("No Documents found.")
        return JSONResponse(content={"message": "No documents found."})

    logger.info(f"Found {len(DOCS)} documents.")

    response = []
    try:
        for d in DOCS:
            score = d[1]
            text = d[0].page_content
            page = d[0].metadata["page"]
            source = d[0].metadata["source"]
            response.append(SearchResponse(text=text, page=page, source=source, score=score))
    except Exception as e:
        for d in DOCS:
            score = d[1]
            text = d[0].page_content
            source = d[0].metadata["source"]
            response.append(SearchResponse(text=text, page=0, source=source, score=score))

    return response


@app.post("/qa")
def question_answer(request: QARequest) -> QAResponse:
    """Answer a question based on the documents in the database.

    Args:
        request (QARequest): The request parameters.

    Raises:
        ValueError: Error if no query or token is provided.

    Returns:
        Tuple: Answer, Prompt and Meta Data
    """
    logger.info("Answering Question")
    # if the query is not provided, raise an error
    if request.query is None:
        raise ValueError("Please provide a Question.")

    token = validate_token(token=request.token, llm_backend=request.llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    # if the history flag is activated and the history is not provided, raise an error
    if request.history and request.history_list is None:
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

        elif request.llm_backend == "gpt4all":
            # summarize the text
            summary = summarize_text_gpt4all(text=text)
            # combine the history and the query
            request.query = f"{summary}\n{request.query}"
        else:
            raise ValueError("Please provide either 'aleph-alpha', 'gpt4all' or 'openai' as a parameter. Other backends are not implemented yet.")

    documents = search_database(
        query=request.query, llm_backend=request.llm_backend, token=token, amount=request.amount, threshold=request.threshold, collection_name=request.collection_name
    )

    # call the qa function
    if request.llm_backend in {"aleph-alpha", "aleph_alpha", "aa"}:
        answer, prompt, meta_data = qa_aleph_alpha(query=request.query, documents=documents, aleph_alpha_token=token)
    elif request.llm_backend == "openai":
        # todo:
        raise ValueError("Please provide either 'aleph-alpha', 'gpt4all' or 'openai' as a parameter. Other backends are not implemented yet.")
    elif request.llm_backend == "gpt4all":
        answer, prompt, meta_data = qa_gpt4all(documents=documents, query=request.query, summarization=request.summarization, language=request.language)
    else:
        raise ValueError("Please provide either 'aleph-alpha', 'gpt4all' or 'openai' as a parameter. Other backends are not implemented yet.")

    return QAResponse(answer=answer, prompt=prompt, meta_data=meta_data)


@app.post("/explanation/explain-qa")
def explain_question_answer(
    query: Optional[str] = None, llm_backend: str = "aa", token: Optional[str] = None, amount: int = 1, threshold: float = 0.0
) -> ExplainQAResponse:
    """Answer a question & explains it based on the documents in the database. This only works with Aleph Alpha.

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
    logger.info("Answering Question and Explaining it.")
    # if the query is not provided, raise an error
    if query is None:
        raise ValueError("Please provide a Question.")

    token = validate_token(token=token, llm_backend=llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    documents = search_database(query=query, llm_backend=llm_backend, token=token, amount=amount, threshold=threshold)

    # call the qa function
    explanation, score, text, answer, meta_data = explain_qa(query=query, document=documents, aleph_alpha_token=token)

    return ExplainQAResponse(explanation=explanation, score=score, text=text, answer=answer, meta_data=meta_data)


@app.post("/explaination/aleph_alpha_explain")
def explain_output(request: ExplainRequest) -> JSONResponse:
    """Explain the output of the question answering system.

    Args:
        request (ExplainRequest): The explain request.

    Raises:
        ValueError: If no token is provided.

    Returns:
        Dict[str, float]: A dictionary containing the prompt and the score of the output.
    """
    logger.info("Explain Output")
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
async def process_document(files: List[UploadFile] = File(...), llm_backend: str = "aa", token: Optional[str] = None, type: str = "invoice") -> None:
    """Process a document.

    Args:
        files (UploadFile): _description_
        llm_backend (str, optional): _description_. Defaults to "openai".
        token (Optional[str], optional): _description_. Defaults to None.
        type (str, optional): _description_. Defaults to "invoice".

    Returns:
        JSONResponse: _description_
    """
    logger.info("Processing Document")
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


def search_database(
    query: str, llm_backend: str = "aa", token: Optional[str] = None, amount: int = 3, threshold: float = 0.0, collection_name: Optional[str] = None
) -> List[tuple[LangchainDocument, float]]:
    """Searches the database for a query.

    Args:
        query (str): The search query.
        llm_backend (str, optional): The LLM provider. Defaults to "openai".
        token (str, optional): The API token. Defaults to None.
        amount (int, optional): The amount of search results. Defaults to 3.

    Raises:
        ValueError: If the LLM provider is not implemented yet.

    Returns:
        JSON List of Documents consisting of the text, page, source and score.
    """
    logger.info("Searching for Documents")
    token = validate_token(token=token, llm_backend=llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    if llm_backend in {"aleph-alpha", "aleph_alpha", "aa"}:
        # Embedd the documents with Aleph Alpha
        documents = search_documents_aleph_alpha(aleph_alpha_token=token, query=query, amount=amount, threshold=threshold, collection_name=collection_name)
    elif llm_backend == "openai":
        documents = search_documents_openai(open_ai_token=token, query=query, amount=amount, threshold=threshold, collection_name=collection_name)
    elif llm_backend == "gpt4all":
        documents = search_documents_gpt4all(query=query, amount=amount, threshold=threshold, collection_name=collection_name)
    else:
        raise ValueError("Please provide either 'aleph-alpha' or 'openai' as a parameter. Other backends are not implemented yet.")

    logger.info(f"Found {len(documents)} documents.")
    return documents


@app.post("/llm/completion/custom")
async def custom_prompt_llm(request: CustomPromptCompletion) -> str:
    """This method sents a custom completion request to the LLM Provider.

    Args:
        request (CustomPromptCompletion): The request parameters.

    Raises:
        ValueError: If the LLM provider is not implemented yet.
    """
    logger.info("Sending Custom Completion Request")
    if request.llm_backend in {"aleph-alpha", "aleph_alpha", "aa"}:
        # sent a completion
        answer = custom_completion_prompt_aleph_alpha(
            prompt=request.prompt,
            token=request.token,
            model=request.model,
            stop_sequences=request.stop_sequences,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    elif request.llm_backend == "OpenAI":
        answer = send_custom_completion_openai(
            prompt=request.prompt,
            token=request.token,
            model=request.model,
            stop_sequences=request.stop_sequences,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    elif request.llm_backend == "gpt4all":
        answer = custom_completion_prompt_gpt4all(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

    else:
        raise ValueError("Please provide either 'aleph-alpha', 'gpt4all' or 'openai' as a parameter. Other backends are not implemented yet.")

    return answer


@app.delete("/embeddings/delete/{llm_provider}/{page}/{source}")
def delete(
    page: int,
    source: str,
    llm_provider: str = "openai",
) -> UpdateResult:
    """Delete a Vector from the database based on the page and source.

    Args:
        page (int): The page of the Document
        source (str): The name of the Document
        llm_provider (str, optional): The LLM Provider. Defaults to "openai".

    Returns:
        UpdateResult: The result of the Deletion Operation from the Vector Database.
    """
    logger.info("Deleting Vector from Database")
    if llm_provider in {"aleph-alpha", "aleph_alpha", "aa"}:
        collection = "aleph-alpha"
    elif llm_provider == "OpenAI":
        collection = "openai"
    elif llm_provider == "GPT4ALL":
        collection = "gpt4all"
    else:
        raise ValueError("Please provide either 'aleph-alpha', 'gpt4all' or 'openai' as a parameter. Other backends are not implemented yet.")

    qdrant_client = load_vec_db_conn()

    result = qdrant_client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.page",
                        match=models.MatchValue(value=page),
                    ),
                    models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source)),
                ],
            )
        ),
    )

    logger.info("Deleted Point from Database via Metadata.")
    return result


@load_config(location="config/db.yml")
def initialize_qdrant_client_config(cfg: DictConfig):
    """Initialize the Qdrant Client.

    Args:
        cfg (DictConfig): Configuration from the file

    Returns:
        _type_: Qdrant Client and Configuration.
    """
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)
    return qdrant_client, cfg


def initialize_aleph_alpha_vector_db() -> None:
    """Initializes the Aleph Alpha vector db.

    Args:
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client, cfg = initialize_qdrant_client_config()
    try:
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_aa)
        logger.info(f"SUCCESS: Collection {cfg.qdrant.collection_name_aa} already exists.")
    except Exception:
        generate_collection_aleph_alpha(qdrant_client, collection_name=cfg.qdrant.collection_name_aa, embeddings_size=cfg.aleph_alpha_embeddings.size)


def generate_collection_aleph_alpha(qdrant_client, collection_name, embeddings_size):
    """Generate a collection for the Aleph Alpha Backend.

    Args:
        qdrant_client (_type_): _description_
        collection_name (_type_): _description_
        embeddings_size (_type_): _description_
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


def initialize_open_ai_vector_db() -> None:
    """Initializes the OpenAI vector db.

    Args:
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client, cfg = initialize_qdrant_client_config()

    try:
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_openai)
        logger.info(f"SUCCESS: Collection {cfg.qdrant.collection_name_openai} already exists.")
    except Exception:
        generate_collection_openai(qdrant_client, collection_name=cfg.qdrant.collection_name_openai)


def generate_collection_openai(qdrant_client, collection_name):
    """Generate a collection for the OpenAI Backend.

    Args:
        qdrant_client (_type_): Qdrant Client Langchain.
        collection_name (_type_): Name of the Collection
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


def initialize_gpt4all_vector_db() -> None:
    """Initializes the GPT4ALL vector db.

    Args:
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client, cfg = initialize_qdrant_client_config()

    try:
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_gpt4all)
        logger.info(f"SUCCESS: Collection {cfg.qdrant.collection_name_gpt4all} already exists.")
    except Exception:
        generate_collection_gpt4all(qdrant_client, collection_name=cfg.qdrant.collection_name_gpt4all)


def generate_collection_gpt4all(qdrant_client, collection_name):
    """Generate a collection for the GPT4ALL Backend.

    Args:
        qdrant_client (Qdrant): Qdrant Client
        collection_name (str): Name of the Collection
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


# initialize the databases
initialize_open_ai_vector_db()
initialize_aleph_alpha_vector_db()
initialize_gpt4all_vector_db()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
