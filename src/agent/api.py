"""Main API."""

import pyfiglet
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from loguru import logger
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent.routes import collection, delete, embeddings, openai_compat, rag, search
from agent.utils.config import Config
from agent.utils.vdb import initialize_all_vector_dbs

load_dotenv(override=True)
config = Config()


initialize_all_vector_dbs(config=config)
logger.info("Startup.")

# configure the Phoenix tracer
tracer_provider = register(
    project_name="rag",
    endpoint=config.phoenix_collector_endpoint,
)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Show startup message
f = pyfiglet.figlet_format("Conv Agent", font="alligator")
logger.info(f"Welcome to\n\n{f}\n\n")


def my_schema() -> dict:
    """Generate the OpenAPI Schema."""
    openapi_schema = get_openapi(
        title="Conversational AI API",
        version="1.0",
        description="Chat with your Documents using Large Language Models.",
        routes=app.routes,
    )
    for path in ("/v1/chat/completions", "/v1/responses"):
        openapi_schema["paths"][path]["post"]["responses"].pop("422", None)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app = FastAPI()
app.openapi = my_schema


def _is_openai_path(path: str) -> bool:
    return path == "/v1" or path.startswith("/v1/")


def _openai_error_response(*, status_code: int, message: str, error_type: str, param: str | None, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


@app.exception_handler(openai_compat.OpenAIAPIError)
async def openai_api_exception_handler(_request: Request, exc: openai_compat.OpenAIAPIError) -> JSONResponse:
    """Render explicit OpenAI compatibility errors."""
    return _openai_error_response(
        status_code=exc.status_code,
        message=exc.message,
        error_type=exc.error_type,
        param=exc.param,
        code=exc.code,
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
    """Use OpenAI error envelopes for HTTP errors below the compatibility prefix."""
    if not _is_openai_path(request.url.path):
        return await http_exception_handler(request, exc)

    return _openai_error_response(
        status_code=exc.status_code,
        message=str(exc.detail),
        error_type="invalid_request_error" if exc.status_code < 500 else "server_error",
        param=None,
        code="not_found" if exc.status_code == 404 else "http_error",
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
    """Use OpenAI validation envelopes only below the compatibility prefix."""
    if not _is_openai_path(request.url.path):
        return await request_validation_exception_handler(request, exc)

    error = exc.errors()[0]
    location = [str(part) for part in error["loc"] if part != "body"]
    param = ".".join(location) or None
    unsupported = error["type"] == "extra_forbidden"
    message = f"Unsupported field: `{param}`." if unsupported else error["msg"]
    return _openai_error_response(
        status_code=400,
        message=message,
        error_type="invalid_request_error",
        param=param,
        code="unsupported_parameter" if unsupported else "invalid_value",
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(f"Global error: {exc}")
    if _is_openai_path(request.url.path):
        return _openai_error_response(
            status_code=500,
            message="Internal server error.",
            error_type="server_error",
            param=None,
            code="internal_error",
        )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "details": str(exc)},
    )


logger.info("Loading REST API Finished.")

app.include_router(router=collection.router, prefix="/collection")
app.include_router(router=embeddings.router, prefix="/embeddings")
app.include_router(router=search.router, prefix="/semantic")
app.include_router(router=rag.router, prefix="/rag")
app.include_router(router=delete.router, prefix="/embeddings")
app.include_router(router=openai_compat.router)


@app.get(path="/", tags=["root"])
def read_root() -> str:
    """Returning the Root."""
    return "Welcome to the RAG Backend. Please navigate to /docs for the OpenAPI!"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
