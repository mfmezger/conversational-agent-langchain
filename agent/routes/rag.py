"""The RAG Routes."""
from fastapi import APIRouter
from loguru import logger

from agent.backend.graph import build_graph
from agent.backend.LLMStrategy import LLMContext, LLMStrategyFactory
from fastapi.responses import StreamingResponse
from agent.data_model.request_data_model import LLMBackend, RAGRequest
from agent.data_model.response_data_model import QAResponse
from agent.utils.utility import combine_text_from_list

graph = build_graph()

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
    model_name = chat_request.model

    async def stream() -> AsyncGenerator:
        documents = []

        async for chunk in graph.with_config(configurable={"model_name": model_name}).astream_events(
            {"retriever_name": model_name, "messages": chat_request.messages}, version="v2"
        ):
            if chunk["event"] == "on_chat_model_stream" and chunk["metadata"]["langgraph_step"] == 2:
                yield json.dumps({"done": False, "content": chunk["data"]["chunk"].content}) + "\n"
            if chunk["name"] == "LangGraph" and chunk["event"] == "on_chain_end":
                documents = [
                    {"document": [doc.page_content], "metadata": [doc.metadata], "source": {"name": doc.metadata["source"]}}
                    for doc in chunk["data"]["output"]["documents"]
                ]
                yield json.dumps({"done": True, "content": "", "citations": documents}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
