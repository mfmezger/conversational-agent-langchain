"""Main API."""

import os
from typing import List
import nltk
from agent.routes import collection, delete, embeddings, rag, search
from agent.utils.vdb import initialize_all_vector_dbs
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from loguru import logger
from phoenix.trace.langchain import LangChainInstrumentor
from agent.utils.utility import check_env_variables


# Load environment variables
load_dotenv(override=True)

# Check for required environment variables
required_env_vars = ['OPENAI_API_KEY', 'COHERE_API_KEY', 'QDRANT_API_KEY']  
check_env_variables(required_env_vars)
logger.info("All necessary Environment variables loaded successfully.")

LangChainInstrumentor().instrument()
nltk.download("punkt")
nltk.download("punkt_tab")


initialize_all_vector_dbs()
logger.info("Vector Database Connection Initialized.")

logger.info("Startup.")

logger.info(
    """

Welcome to

 ██████╗ ██████╗ ███╗   ██╗██╗   ██╗███████╗██████╗ ███████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗ █████╗ ██╗          █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔════╝██╔═══██╗████╗  ██║██║   ██║██╔════╝██╔══██╗██╔════╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔══██╗██║         ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██║     ██║   ██║██╔██╗ ██║██║   ██║█████╗  ██████╔╝███████╗███████║   ██║   ██║██║   ██║██╔██╗ ██║███████║██║         ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
██║     ██║   ██║██║╚██╗██║╚██╗ ██╔╝██╔══╝  ██╔══██╗╚════██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║██╔══██║██║         ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
╚██████╗╚██████╔╝██║ ╚████║ ╚████╔╝ ███████╗██║  ██║███████║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║██║  ██║███████╗    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝



"""
)


def my_schema() -> dict:
    """Generate the OpenAPI Schema."""
    openapi_schema = get_openapi(
        title="Conversational AI API",
        version="1.0",
        description="Chat with your Documents using Conversational AI by Aleph Alpha, GPT4ALL and OpenAI.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app = FastAPI(debug=True)
app.openapi = my_schema


app.include_router(collection.router, prefix="/collection")
app.include_router(embeddings.router, prefix="/embeddings")
app.include_router(search.router, prefix="/semantic")
app.include_router(rag.router, prefix="/rag")
app.include_router(delete.router, prefix="/embeddings")


@app.get("/", tags=["root"])
def read_root() -> str:
    """Returning the Root."""
    return "Welcome to the RAG Backend. Please navigate to /docs for the OpenAPI!"




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
