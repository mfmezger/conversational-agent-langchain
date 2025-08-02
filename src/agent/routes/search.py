"""The search routes."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger

from agent.data_model.request_data_model import SearchParams
from agent.data_model.response_data_model import SearchResponse
from agent.utils.retriever import get_retriever

router = APIRouter()


@router.post("/search", tags=["search"], response_model=list[SearchResponse])
async def search(search: SearchParams) -> list[SearchResponse] | JSONResponse:
    """Search for documents."""
    logger.info("Searching for Documents")
    retriever = get_retriever(
        collection_name=search.collection_name,
        k=search.k,
    )
    docs = await retriever.ainvoke(search.query)

    if not docs:
        logger.info("No Documents found.")
        return JSONResponse(content={"message": "No documents found."})

    logger.info(f"Found {len(docs)} documents.")
    return [SearchResponse(text=d.page_content, page=d.metadata["page"], source=d.metadata["source"]) for d in docs]
