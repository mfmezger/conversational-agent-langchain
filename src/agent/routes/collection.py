"""Routes for the collection management."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agent.backend.services.embedding_management import EmbeddingManagement

router = APIRouter()


@router.post("/create/{collection_name}", tags=["collection"])
def create_collection(collection_name: str) -> JSONResponse:
    """Create a new collection.

    Args:
    ----
        llm_provider (LLMProvider): The LLM provider.
        collection_name (str): Name of the Qdrant Collection

    Returns:
    -------
        JSONResponse: Success Message.

    """
    service = EmbeddingManagement(collection_name=collection_name)
    service.createe_collection_collection(name=collection_name)
    return JSONResponse(content={"message": f"Collection {collection_name} created."})
