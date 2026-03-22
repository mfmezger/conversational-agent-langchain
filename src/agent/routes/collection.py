"""Routes for the collection management."""

from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from agent.utils.vdb import initialize_vector_db_async

router = APIRouter()


@router.post(path="/create/{collection_name}", tags=["collection"])
async def create_collection(collection_name: str, embeddings_size: Annotated[int, Query(gt=0, le=5000)]) -> JSONResponse:
    """Create a new collection.

    Args:
    ----
        collection_name (str): Name of the Qdrant Collection
        embeddings_size (int): The size of the embeddings.

    Returns:
    -------
        JSONResponse: Success Message.

    """
    await initialize_vector_db_async(collection_name=collection_name, embeddings_size=embeddings_size)
    return JSONResponse(content={"message": f"Collection {collection_name} created."})
