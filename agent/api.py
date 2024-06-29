"""FastAPI Backend for the Knowledge Agent."""
from pathlib import Path

import nltk
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.openapi.utils import get_openapi
from loguru import logger
from phoenix.trace.langchain import LangChainInstrumentor
from qdrant_client import models
from qdrant_client.http.models.models import UpdateResult
from starlette.responses import JSONResponse

from agent.backend.LLMStrategy import LLMContext, LLMStrategyFactory
from agent.data_model.request_data_model import (
    CustomPromptCompletion,
    EmbeddTextRequest,
    ExplainQARequest,
    LLMBackend,
    LLMProvider,
    RAGRequest,
    SearchParams,
)
from agent.data_model.response_data_model import (
    EmbeddingResponse,
    ExplainQAResponse,
    QAResponse,
    SearchResponse,
)
from agent.utils.utility import (
    combine_text_from_list,
    create_tmp_folder,
)
from agent.utils.vdb import initialize_all_vector_dbs, load_vec_db_conn

LangChainInstrumentor().instrument()
nltk.download("punkt")
# add file logger for loguru
# logger.add("logs/file_{time}.log", backtrace=False, diagnose=False)
logger.info("Startup.")


def my_schema() -> dict:
    """Used to generate the OpenAPI schema.

    Returns
    -------
        FastAPI: FastAPI App
    """
    openapi_schema = get_openapi(
        title="Conversational AI API",
        version="1.0",
        description="Chat with your Documents using Conversational AI by Aleph Alpha, GPT4ALL and OpenAI.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# initialize the Fast API Application.
app = FastAPI(debug=True)
app.openapi = my_schema

load_dotenv()

# load the token from the environment variables, is None if not set.
logger.info("Loading REST API Finished.")


@app.get("/", tags=["root"])
def read_root() -> str:
    """Returns the welcome message.

    Returns
    -------
        str: The welcome message.

    """
    return "Welcome to the RAG Backend. Please navigate to /docs for the OpenAPI!"


@app.post("/collection/create/{llm_provider}/{collection_name}", tags=["collection"])
def create_collection(llm_provider: LLMProvider, collection_name: str) -> JSONResponse:
    """Create a new collection in the vector database.

    Args:
    ----
        llm_provider (LLMProvider): Name of the LLM Provider
        collection_name (str): Name of the Collection

    """
    service = LLMContext(LLMStrategyFactory.get_strategy(strategy_type=llm_provider, token="", collection_name=collection_name))

    service.create_collection(name=collection_name)

    # return a success message
    return JSONResponse(content={"message": f"Collection {collection_name} created."})


@app.post("/embeddings/documents", tags=["embeddings"])
async def post_embedd_documents(llm_backend: LLMBackend, files: list[UploadFile] = File(...), file_ending: str = ".pdf") -> EmbeddingResponse:
    """Uploads multiple documents to the backend. Can be.

    Args:
    ----
        llm_backend (LLMBackend): The LLM Backend.
        files (List[UploadFile], optional): Uploaded files. Defaults to File(...).
        file_ending (str, optional): The file ending of the documents. Defaults to ".pdf". Can also be ".txt".

    Returns:
    -------
        JSONResponse: The response as JSON.

    """
    logger.info("Embedding Multiple Documents")

    tmp_dir = create_tmp_folder()

    service = LLMContext(LLMStrategyFactory.get_strategy(strategy_type=LLMProvider.ALEPH_ALPHA, collection_name=llm_backend.collection_name))

    file_names = []

    for file in files:
        file_name = file.filename
        file_names.append(file_name)

        # Save the file to the temporary folder
        if tmp_dir is None or not Path(tmp_dir).exists():
            msg = "Please provide a temporary folder to save the files."
            raise ValueError(msg)

        if file_name is None:
            msg = "Please provide a file to save."
            raise ValueError(msg)

        with Path(tmp_dir / file_name).open("wb") as f:
            f.write(await file.read())

    service.embed_documents(folder=tmp_dir, file_ending=file_ending)

    return EmbeddingResponse(status="success", files=file_names)


@app.post("/embeddings/string/", tags=["embeddings"])
async def embedd_text(embedding: EmbeddTextRequest, llm_backend: LLMBackend) -> EmbeddingResponse:
    """Embeds a string in the database.

    Args:
    ----
        embedding (EmbeddTextRequest): The request parameters.
        llm_backend (LLMBackend): The LLM Backend.

    Raises:
    ------
        ValueError: If no token is provided or if no LLM provider is specified.

    Returns:
    -------
        JSONResponse: A response indicating that the text was received and saved, along with the name of the file it was saved to.

    """
    logger.info("Embedding Text")

    service = LLMContext(LLMStrategyFactory.get_strategy(strategy_type=llm_backend.llm_provider, collection_name=llm_backend.collection_name))

    # save the string to a txt file in a uuid directory
    tmp_dir = create_tmp_folder()
    with (Path(tmp_dir) / (embedding.file_name + ".txt")).open("w") as f:
        f.write(embedding.text)
    service.embed_documents(directory=tmp_dir, file_ending=".txt")

    return EmbeddingResponse(status="success", files=[embedding.file_name])


@app.post("/semantic/search", tags=["search"])
def search(search: SearchParams, llm_backend: LLMBackend) -> list[SearchResponse]:
    """Searches for a query in the vector database.

    Args:
    ----
        search (SearchRequest): The search request.
        llm_backend (LLMBackend): The LLM Backend.

    Raises:
    ------
        ValueError: If the LLM provider is not implemented yet.

    Returns:
    -------
        List[str]: A list of matching documents.

    """
    logger.info("Searching for Documents")

    service = LLMContext(LLMStrategyFactory.get_strategy(strategy_type=llm_backend.llm_provider, token=llm_backend.token, collection_name=llm_backend.collection_name))

    docs = service.search(search=search)

    if not docs:
        logger.info("No Documents found.")
        return JSONResponse(content={"message": "No documents found."})

    logger.info(f"Found {len(docs)} documents.")

    response = []
    for d in docs:
        score = d[1]
        text = d[0].page_content
        page = d[0].metadata["page"]
        source = d[0].metadata["source"]
        response.append(SearchResponse(text=text, page=page, source=source, score=score))

    return response


@app.post("/rag", tags=["rag"])
def question_answer(rag: RAGRequest, llm_backend: LLMBackend) -> QAResponse:
    """Answer a question based on the documents in the database.

    Args:
    ----
        rag (RAGRequest): The request parameters.
        llm_backend (LLMBackend): The LLM Backend.

    Raises:
    ------
        ValueError: Error if no query or token is provided.

    Returns:
    -------
        Tuple: Answer, Prompt and Meta Data

    """
    logger.info("Answering Question")
    # if the query is not provided, raise an error
    if rag.search.query is None:
        msg = "Please provide a Question."
        raise ValueError(msg)

    service = LLMContext(LLMStrategyFactory.get_strategy(strategy_type=llm_backend.llm_provider, collection_name=llm_backend.collection_name))
    # summarize the history
    if rag.history:
        # combine the texts
        # TODO: adopt to dict
        text = combine_text_from_list(rag.history)
        service.summarize_text(text=text, token="")

    rag_chain = service.create_rag_chain(rag=rag, llm_backend=llm_backend)

    chain_result = rag_chain.invoke(rag.query)

    return QAResponse(answer=chain_result["answer"], prompt=prompt, meta_data=chain_result["meta_data"])


# TODO: implement server side events.
@app.post("/rag/stream", tags=["rag"])
def question_answer_stream(rag: RAGRequest, llm_backend: LLMBackend) -> None:
    pass


@app.post("/explanation/explain-qa", tags=["explanation"])
def explain_question_answer(explain_request: ExplainQARequest, llm_backend: LLMBackend) -> ExplainQAResponse:
    """Answer a question & explains it based on the documents in the database. This only works with Aleph Alpha.

    This uses the normal qa but combines it with the explain function.

    Args:
    ----
        explain_request (ExplainQARequest): The Explain Requesat
        llm_backend (LLMBackend): The LLM Backend.

    Raises:
    ------
        ValueError: Error if no query or token is provided.

    Returns:
    -------
        Tuple: Answer, Prompt and Meta Data

    """
    logger.info("Answering Question and Explaining it.")
    # if the query is not provided, raise an error
    if explain_request.rag_request.search.query is None:
        msg = "Please provide a Question."
        raise ValueError(msg)

    service = LLMContext(
        LLMStrategyFactory.get_strategy(strategy_type=llm_backend.llm_provider, token=search.llm_backend.token, collection_name=llm_backend.collection_name)
    )

    documents = service.search(explain_request.rag_request.search)

    # call the qa function
    explanation, score, text, answer, meta_data = service.explain_qa(
        query=explain_request.rag_request.search.query,
        explain_threshold=explain_request.explain_threshold,
        document=documents,
        aleph_alpha_token=explain_request.rag_request.search.llm_backend.token,
    )

    return ExplainQAResponse(explanation=explanation, score=score, text=text, answer=answer, meta_data=meta_data)


# @app.post("/process_document", tags=["custom"])
# async def process_document(files: list[UploadFile] = File(...), llm_backend: str = "aa", token: str | None = None, document_type: str = "invoice") -> None:
#     """Process a document.

#     Args:
#     ----
#         files (UploadFile): _description_
#         llm_backend (str, optional): _description_. Defaults to "openai".
#         token (Optional[str], optional): _description_. Defaults to None.
#         type (str, optional): _description_. Defaults to "invoice".

#     Returns:
#     -------
#         JSONResponse: _description_
#     """
#     logger.info("Processing Document")

#     # Create a temporary folder to save the files
#     tmp_dir = create_tmp_folder()

#     file_names = []

#     for file in files:
#         file_name = file.filename
#         file_names.append(file_name)

#         # Save the file to the temporary folder
#         if tmp_dir is None or not Path(tmp_dir).exists():
#             msg = "Please provide a temporary folder to save the files."
#             raise ValueError(msg)

#         if file_name is None:
#             msg = "Please provide a file to save."
#             raise ValueError(msg)

#         with Path(tmp_dir / file_name).open() as f:
#             f.write(await file.read())

#     process_documents_aleph_alpha(folder=tmp_dir, , type=document_type)

#     logger.info(f"Found {len(documents)} documents.")
#     return documents


@app.post("/llm/completion/custom", tags=["custom"])
async def custom_prompt_llm(request: CustomPromptCompletion) -> str:
    """The method sents a custom completion request to the LLM Provider.

    Args:
    ----
        request (CustomPromptCompletion): The request parameters.

    Raises:
    ------
        ValueError: If the LLM provider is not implemented yet.

    """
    logger.info("Sending Custom Completion Request")

    service = LLMContext(
        LLMStrategyFactory.get_strategy(
            strategy_type=request.search.llm_backend.llm_provider, token=request.search.llm_backend.token, collection_name=request.search.collection_name
        )
    )

    return service.generate(request.text)


@app.delete("/embeddings/delete/{llm_provider}/{page}/{source}", tags=["embeddings"])
def delete(
    page: int,
    source: str,
    llm_provider: LLMProvider = LLMProvider.OPENAI,
) -> UpdateResult:
    """Delete a Vector from the database based on the page and source.

    Args:
    ----
        page (int): The page of the Document
        source (str): The name of the Document
        llm_provider (LLMProvider, optional): The LLM Provider. Defaults to LLMProvider.OPENAI.

    Returns:
    -------
        UpdateResult: The result of the Deletion Operation from the Vector Database.
    """
    logger.info("Deleting Vector from Database")
    if llm_provider == LLMProvider.ALEPH_ALPHA:
        collection = "aleph-alpha"
    elif llm_provider == LLMProvider.OPENAI:
        collection = "openai"
    elif llm_provider == LLMProvider.GPT4ALL:
        collection = "gpt4all"
    else:
        msg = f"Unsupported LLM provider: {llm_provider}"
        raise ValueError(msg)

    qdrant_client, _ = load_vec_db_conn()

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


# initialize the databases
initialize_all_vector_dbs()

# for debugging useful.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
