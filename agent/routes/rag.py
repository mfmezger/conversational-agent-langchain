"""The RAG Routes."""
from fastapi import APIRouter
from loguru import logger

from agent.backend.LLMStrategy import LLMContext, LLMStrategyFactory
from agent.data_model.request_data_model import LLMBackend, RAGRequest
from agent.data_model.response_data_model import QAResponse
from agent.utils.utility import combine_text_from_list

router = APIRouter()


@router.post("/", tags=["rag"])
def question_answer(rag: RAGRequest, llm_backend: LLMBackend) -> QAResponse:
    """Answering the Question."""
    logger.info("Answering Question")
    if rag.search.query is None:
        msg = "Please provide a Question."
        raise ValueError(msg)
    service = LLMContext(LLMStrategyFactory.get_strategy(strategy_type=llm_backend.llm_provider, collection_name=llm_backend.collection_name))
    if rag.history:
        text = combine_text_from_list(rag.history)
        service.summarize_text(text=text, token="")
    rag_chain = service.create_rag_chain(rag=rag, llm_backend=llm_backend)
    chain_result = rag_chain.invoke(rag.query)
    return QAResponse(answer=chain_result["answer"], prompt=chain_result["prompt"], meta_data=chain_result["meta_data"])


@router.post("/stream", tags=["rag"])
def question_answer_stream(rag: RAGRequest, llm_backend: LLMBackend) -> None:
    """Stream the Answering."""
