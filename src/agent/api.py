"""Main API."""

import pyfiglet
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from loguru import logger
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

from agent.routes import collection, delete, embeddings, rag, search
from agent.utils.config import Config
from agent.utils.vdb import initialize_all_vector_dbs

load_dotenv(override=True)
config = Config()


initialize_all_vector_dbs(config=config)
logger.info("Startup.")

# configure the Phoenix tracer
tracer_provider = register(
    project_name="rag",
)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Show startup message
f = pyfiglet.figlet_format("Conv. Agent", font="univers")
logger.info(f"Welcome to {f}")


def my_schema() -> dict:
    """Generate the OpenAPI Schema."""
    openapi_schema = get_openapi(
        title="Conversational AI API",
        version="1.0",
        description="Chat with your Documents using Large Language Models.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app = FastAPI(debug=True)
app.openapi = my_schema

logger.info("Loading REST API Finished.")

app.include_router(router=collection.router, prefix="/collection")
app.include_router(router=embeddings.router, prefix="/embeddings")
app.include_router(router=search.router, prefix="/semantic")
app.include_router(router=rag.router, prefix="/rag")
app.include_router(router=delete.router, prefix="/embeddings")


@app.get("/", tags=["root"])
def read_root() -> str:
    """Returning the Root."""
    return "Welcome to the RAG Backend. Please navigate to /docs for the OpenAPI!"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
