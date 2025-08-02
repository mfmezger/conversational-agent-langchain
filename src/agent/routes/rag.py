"""The RAG Routes."""

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agent.backend.graph import Graph
from agent.data_model.request_data_model import RAGRequest
from agent.data_model.response_data_model import QAResponse

graph = Graph().build_graph()


router = APIRouter()


@router.post("/", tags=["rag"])
async def question_answer(rag: RAGRequest) -> QAResponse:
    """Answering the Question."""
    messages = [dict(m) for m in rag.messages]
    chain_result = await graph.with_config({"collection_name": rag.collection_name}).ainvoke({"messages": messages})

    documents = [{"document": [doc.page_content], "metadata": [doc.metadata]} for doc in chain_result["documents"]]
    return QAResponse(answer=chain_result["messages"][-1].content, meta_data=documents)


@router.post("/stream", tags=["rag"])
async def question_answer_stream(rag: RAGRequest) -> StreamingResponse:
    """Stream the Answering."""
    messages = [dict(m) for m in rag.messages]

    async def stream() -> AsyncGenerator:
        documents = []

        async for chunk in graph.with_config({"collection_name": rag.collection_name}).astream_events({"messages": messages}, version="v2"):
            if chunk["event"] == "on_chat_model_stream" and chunk["metadata"]["langgraph_step"] == 2:
                yield json.dumps({"done": False, "content": chunk["data"]["chunk"].content}) + "\n"
            if chunk["name"] == "LangGraph" and chunk["event"] == "on_chain_end":
                documents = [{"document": [doc.page_content], "metadata": [doc.metadata]} for doc in chunk["data"]["output"]["documents"]]
                yield json.dumps({"done": True, "content": "", "citations": documents}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
