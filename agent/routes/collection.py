"""Routes for the collection management."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agent.backend.LLMStrategy import LLMContext, LLMStrategyFactory
from agent.data_model.request_data_model import LLMProvider

router = APIRouter()


@router.post("/create/{llm_provider}/{collection_name}", tags=["collection"])
def create_collection(llm_provider: LLMProvider, collection_name: str) -> JSONResponse:
    """Create a new collection.

    Args:
    ----
        llm_provider (LLMProvider): The LLM provider.
        collection_name (str): Name of the Qdrant Collection

    Returns:
    -------
        JSONResponse: Success Message.
    """
    service = LLMContext(LLMStrategyFactory.get_strategy(strategy_type=llm_provider, token="", collection_name=collection_name))
    service.create_collection(name=collection_name)
    return JSONResponse(content={"message": f"Collection {collection_name} created."})
