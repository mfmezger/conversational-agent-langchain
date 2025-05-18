"""Routes for the collection management."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agent.utils.vdb import initialize_vector_db

router = APIRouter()


@router.post(path="/create/{collection_name}", tags=["collection"])
def create_collection(collection_name: str, embeddings_size: int) -> JSONResponse:
    """Create a new collection.

    Args:
    ----
        collection_name (str): Name of the Qdrant Collection
        embeddings_size (int): The size of the embeddings.

    Returns:
    -------
        JSONResponse: Success Message.

    """
    initialize_vector_db(collection_name=collection_name, embeddings_size=embeddings_size)
    return JSONResponse(content={"message": f"Collection {collection_name} created."})
