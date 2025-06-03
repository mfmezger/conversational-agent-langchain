"""Route to handle the delection of a vector from the database."""

from fastapi import APIRouter
from loguru import logger
from qdrant_client import models
from qdrant_client.http.models.models import UpdateResult

from agent.utils.vdb import load_vec_db_conn

router = APIRouter()


@router.delete("/delete/{page}/{source}", tags=["embeddings"])
def delete(page: int, source: str, collection_name: str) -> UpdateResult:
    """Delete a vector from the database.

    Args:
    ----
        page (int): Number of the page in the document
        source (str): Name of the Document
        collection_name (str): Large Language Model Provider. Defaults to LLMProvider.OPENAI.

    Raises:
    ------
        ValueError: Wrong LLM Provider

    Returns:
    -------
        UpdateResult: Result of the Update.

    """
    logger.info("Deleting Vector from Database")
    qdrant_client, _ = load_vec_db_conn()
    result = qdrant_client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="metadata.page", match=models.MatchValue(value=page)),
                    models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source)),
                ],
            )
        ),
    )
    logger.info("Deleted Point from Database via Metadata.")
    return result
