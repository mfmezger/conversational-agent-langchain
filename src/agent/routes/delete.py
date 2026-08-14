"""Route to handle the deletion of a vector from the database."""

from fastapi import APIRouter
from qdrant_client.http.models.models import UpdateResult

from agent.utils.vdb import delete_documents_by_source

router = APIRouter()


@router.delete("/delete/{source}", tags=["embeddings"])
async def delete(source: str, collection_name: str) -> UpdateResult:
    """Delete a vector from the database.

    Args:
    ----
        source (str): Name of the Document
        collection_name (str): Name of the Qdrant Collection.

    Returns:
    -------
        UpdateResult: Result of the Update.

    """
    return await delete_documents_by_source(collection_name=collection_name, source=source)
