"""The RAG Routes."""

import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agent.backend.graph import Graph
from agent.data_model.request_data_model import OpenAIChatRequest, RAGRequest
from agent.data_model.response_data_model import QAResponse

graph = Graph().build_graph()


router = APIRouter()


@router.post("/", tags=["rag"])
async def question_answer(rag: RAGRequest) -> QAResponse:
    """Answering the Question."""
    messages = [dict(m) for m in rag.messages]
    chain_result = await graph.with_config({"metadata": {"collection_name": rag.collection_name}}).ainvoke({"messages": messages})

    documents = [{"document": [doc.page_content], "metadata": [doc.metadata]} for doc in chain_result["documents"]]
    return QAResponse(answer=chain_result["messages"][-1].content, meta_data=documents)


@router.post("/chat/completions", tags=["rag"], response_model=None)
async def openai_chat_completions(request: OpenAIChatRequest) -> dict | StreamingResponse:
    """OpenAI compatible chat completions endpoint."""
    messages = [dict(m) for m in request.messages]
    # Use the model as collection_name or "default"
    collection_name = request.model if request.model else "default"

    # Generate an ID for the response
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    if not request.stream:
        chain_result = await graph.with_config({"metadata": {"collection_name": collection_name}}).ainvoke({"messages": messages})
        answer = chain_result["messages"][-1].content

        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    async def openai_stream() -> AsyncGenerator:
        # Send the first chunk with the role
        initial_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ]
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        async for chunk in graph.with_config({"metadata": {"collection_name": collection_name}}).astream_events({"messages": messages}, version="v2"):
            if chunk["event"] == "on_chat_model_stream" and chunk["metadata"].get("langgraph_node") in ["response_synthesizer", "response_synthesizer_cohere"]:
                content = chunk["data"]["chunk"].content
                if content:
                    delta_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }
                        ]
                    }
                    yield f"data: {json.dumps(delta_chunk)}\n\n"

        # Send the final chunk
        final_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(openai_stream(), media_type="text/event-stream")


@router.post("/stream", tags=["rag"])
async def question_answer_stream(rag: RAGRequest) -> StreamingResponse:
    """Stream the Answering."""
    messages = [dict(m) for m in rag.messages]

    async def stream() -> AsyncGenerator:
        documents = []

        # Yield initial status
        yield json.dumps({"type": "status", "data": "Starting request..."}) + "\n"

        async for chunk in graph.with_config({"metadata": {"collection_name": rag.collection_name}}).astream_events({"messages": messages}, version="v2"):
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
