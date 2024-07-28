"""Main API."""

import nltk
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from loguru import logger
from phoenix.trace.langchain import LangChainInstrumentor

from agent.routes import (collection, delete, embeddings, explanation, rag,
                          search)
from agent.utils.vdb import initialize_all_vector_dbs

LangChainInstrumentor().instrument()
nltk.download("punkt")
logger.info("Startup.")

logger.info("""

Welcome to 

 ██████╗ ██████╗ ███╗   ██╗██╗   ██╗███████╗██████╗ ███████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗ █████╗ ██╗          █████╗  ██████╗ ███████╗███╗   ██╗████████╗    
██╔════╝██╔═══██╗████╗  ██║██║   ██║██╔════╝██╔══██╗██╔════╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔══██╗██║         ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝    
██║     ██║   ██║██╔██╗ ██║██║   ██║█████╗  ██████╔╝███████╗███████║   ██║   ██║██║   ██║██╔██╗ ██║███████║██║         ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║       
██║     ██║   ██║██║╚██╗██║╚██╗ ██╔╝██╔══╝  ██╔══██╗╚════██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║██╔══██║██║         ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║       
╚██████╗╚██████╔╝██║ ╚████║ ╚████╔╝ ███████╗██║  ██║███████║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║██║  ██║███████╗    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║       
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝       
                                                                                                                                                                       


""")

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

load_dotenv(override=True)
logger.info("Loading REST API Finished.")

app.include_router(collection.router, prefix="/collection")
app.include_router(embeddings.router, prefix="/embeddings")
app.include_router(search.router, prefix="/semantic")
app.include_router(rag.router, prefix="/rag")
app.include_router(explanation.router, prefix="/explanation")
app.include_router(delete.router, prefix="/embeddings")


@app.get("/", tags=["root"])
def read_root() -> str:
    """Returning the Root."""
    return "Welcome to the RAG Backend. Please navigate to /docs for the OpenAPI!"


# initialize the databases
initialize_all_vector_dbs()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
