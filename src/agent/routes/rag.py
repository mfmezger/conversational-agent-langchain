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
    chain_result = await graph.with_config({"metadata": {"collection_name": rag.collection_name}}).ainvoke(
        {
            "messages": messages,
            "user_id": rag.user_id,
            "session_id": rag.session_id,
            "agent_id": rag.agent_id,
        }
    )

    documents = [{"document": [doc.page_content], "metadata": [doc.metadata]} for doc in chain_result["documents"]]
    return QAResponse(answer=chain_result["messages"][-1].content, meta_data=documents)


@router.post("/stream", tags=["rag"])
async def question_answer_stream(rag: RAGRequest) -> StreamingResponse:
    """Stream the Answering."""
    messages = [dict(m) for m in rag.messages]
    initial_state = {
        "messages": messages,
        "user_id": rag.user_id,
        "session_id": rag.session_id,
        "agent_id": rag.agent_id,
    }

    async def stream() -> AsyncGenerator:
        documents = []

        # Yield initial status
        yield json.dumps({"type": "status", "data": "Starting request..."}) + "\n"

        async for chunk in graph.with_config({"metadata": {"collection_name": rag.collection_name}}).astream_events(initial_state, version="v2"):
            # Status updates for Retrieval
            if chunk["event"] == "on_chain_start" and chunk["name"] in ["retriever", "retriever_with_chat_history"]:
                yield json.dumps({"type": "status", "data": "Searching documents..."}) + "\n"

            elif chunk["event"] == "on_chain_end" and chunk["name"] in ["retriever", "retriever_with_chat_history"]:
                num_docs = len(chunk["data"]["output"].get("documents", []))
                yield json.dumps({"type": "status", "data": f"Found {num_docs} documents."}) + "\n"

            # Status updates for Generation
            elif chunk["event"] == "on_chat_model_start":
                yield json.dumps({"type": "status", "data": "Generating answer..."}) + "\n"

            # Content streaming
            elif chunk["event"] == "on_chat_model_stream" and chunk["metadata"].get("langgraph_node") in ["response_synthesizer", "response_synthesizer_cohere"]:
                content = chunk["data"]["chunk"].content
                if content:
                    yield json.dumps({"type": "content", "data": content}) + "\n"

            # Final Citations
            elif chunk["name"] == "LangGraph" and chunk["event"] == "on_chain_end" and "documents" in chunk["data"]["output"]:
                documents = [{"document": [doc.page_content], "metadata": [doc.metadata]} for doc in chunk["data"]["output"]["documents"]]
                yield json.dumps({"type": "citation", "data": documents}) + "\n"

        # Done event
        yield json.dumps({"type": "status", "data": "Done."}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
