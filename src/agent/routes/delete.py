"""Route to handle the deletion of a vector from the database."""

from fastapi import APIRouter
from qdrant_client.http.models.models import UpdateResult

from agent.dependencies import VDBResourcesDep
from agent.utils.vdb import delete_documents_by_source

router = APIRouter()


@router.delete("/delete/{source}", tags=["embeddings"])
async def delete(source: str, collection_name: str, resources: VDBResourcesDep) -> UpdateResult:
    """Delete a vector from the database.

    Args:
    ----
        source (str): Name of the Document
        collection_name (str): Name of the Qdrant Collection.
        resources (VDBResources): Application-owned vector database resources.

    Returns:
    -------
        UpdateResult: Result of the Update.

    """
    return await delete_documents_by_source(
        client=resources.async_client,
        collection_name=collection_name,
        source=source,
    )
