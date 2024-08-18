"""The RAG Routes."""

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agent.backend.graph import build_graph
from agent.data_model.request_data_model import LLMBackend, RAGRequest
from agent.data_model.response_data_model import QAResponse

graph = build_graph()

router = APIRouter()


@router.post("/", tags=["rag"])
def question_answer(rag: RAGRequest, llm_backend: LLMBackend) -> QAResponse:
    """Answering the Question."""
    messages = [dict(m) for m in rag.messages]
    chain_result = graph.with_config(configurable={"model_name": llm_backend.llm_provider}).invoke({"retriever_name": llm_backend.llm_provider, "messages": messages})

    documents = [{"document": [doc.page_content], "metadata": [doc.metadata]} for doc in chain_result["documents"]]
    return QAResponse(answer=chain_result["messages"][-1].content, meta_data=documents)


@router.post("/stream", tags=["rag"])
def question_answer_stream(rag: RAGRequest, llm_backend: LLMBackend) -> None:
    """Stream the Answering."""
    model_name = llm_backend.llm_provider
    messages = [dict(m) for m in rag.messages]

    async def stream() -> AsyncGenerator:
        documents = []

        async for chunk in graph.with_config(configurable={"model_name": model_name}).astream_events({"retriever_name": model_name, "messages": messages}, version="v2"):
            if chunk["event"] == "on_chat_model_stream" and chunk["metadata"]["langgraph_step"] == 2:
                yield json.dumps({"done": False, "content": chunk["data"]["chunk"].content}) + "\n"
            if chunk["name"] == "LangGraph" and chunk["event"] == "on_chain_end":
                documents = [{"document": [doc.page_content], "metadata": [doc.metadata]} for doc in chunk["data"]["output"]["documents"]]
                yield json.dumps({"done": True, "content": "", "citations": documents}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
