"""The Route to handle Explanations."""

from fastapi import APIRouter
from loguru import logger

from agent.backend.LLMStrategy import LLMContext, LLMStrategyFactory
from agent.data_model.request_data_model import ExplainQARequest, LLMBackend
from agent.data_model.response_data_model import ExplainQAResponse

router = APIRouter()


@router.post("/explain-qa", tags=["explanation"])
def explain_question_answer(explain_request: ExplainQARequest, llm_backend: LLMBackend) -> ExplainQAResponse:
    """Aleph Alpha Explanation for the Question Answering."""
    logger.info("Answering Question and Explaining it.")
    if explain_request.rag_request.search.query is None:
        msg = "Please provide a Question."
        raise ValueError(msg)
    service = LLMContext(LLMStrategyFactory.get_strategy(strategy_type=llm_backend.llm_provider, token=llm_backend.token, collection_name=llm_backend.collection_name))
    documents = service.search(explain_request.rag_request.search)
    explanation, score, text, answer, meta_data = service.explain_qa(
        query=explain_request.rag_request.search.query,
        explain_threshold=explain_request.explain_threshold,
        document=documents,
        aleph_alpha_token=explain_request.rag_request.search.llm_backend.token,
    )
    return ExplainQAResponse(explanation=explanation, score=score, text=text, answer=answer, meta_data=meta_data)
