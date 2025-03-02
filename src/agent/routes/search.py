"""The search routes."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger

from agent.backend.services.embedding_management import EmbeddingManagement
from agent.data_model.request_data_model import SearchParams
from agent.data_model.response_data_model import SearchResponse

router = APIRouter()


@router.post("/search", tags=["search"])
def search(search: SearchParams, collection_name: str) -> list[SearchResponse]:
    """Search for documents."""
    logger.info("Searching for Documents")
    service = EmbeddingManagement(collection_name=collection_name)
    docs = service.search(search=search)

    if not docs:
        logger.info("No Documents found.")
        return JSONResponse(content={"message": "No documents found."})

    logger.info(f"Found {len(docs)} documents.")
    return [SearchResponse(text=d[0].page_content, page=d[0].metadata["page"], source=d[0].metadata["source"], score=d[1]) for d in docs]
