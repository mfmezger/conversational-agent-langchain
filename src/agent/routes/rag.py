"""The RAG Routes."""

import logging
from collections.abc import AsyncIterable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agent.backend.graph import Graph
from agent.data_model.request_data_model import RAGRequest
from agent.data_model.response_data_model import (
    CitationDocument,
    QAResponse,
    StreamCitationEvent,
    StreamContentEvent,
    StreamErrorEvent,
    StreamStatusEvent,
)

logger = logging.getLogger(__name__)

graph = Graph().build_graph()


router = APIRouter()


STREAM_EVENT_SCHEMA = {
    "oneOf": [
        StreamStatusEvent.model_json_schema(),
        StreamContentEvent.model_json_schema(),
        StreamCitationEvent.model_json_schema(),
        StreamErrorEvent.model_json_schema(),
    ]
}

STREAM_EVENT_EXAMPLES = {
    "status": {
        "summary": "Status event",
        "value": {"type": "status", "data": "Generating answer..."},
    },
    "content": {
        "summary": "Content event",
        "value": {"type": "content", "data": "chunk1"},
    },
    "citation": {
        "summary": "Citation event",
        "value": {
            "type": "citation",
            "data": [CitationDocument(document=["doc1"], metadata=[{"source": "test"}]).model_dump()],
        },
    },
    "error": {
        "summary": "Error event",
        "value": {"type": "error", "data": "An internal error occurred during streaming."},
    },
}


class NDJSONStreamingResponse(StreamingResponse):
    """Streaming response with NDJSON media type."""

    media_type = "application/x-ndjson"


def _ndjson_event(
    event: StreamStatusEvent | StreamContentEvent | StreamCitationEvent | StreamErrorEvent,
) -> str:
    """Encode one NDJSON stream event."""
    return event.model_dump_json() + "\n"


def _stream_event_from_chunk(chunk: dict) -> str | None:
    """Map a LangGraph chunk to one NDJSON event."""
    event = None

    if chunk["event"] == "on_chain_start" and chunk["name"] in ["retriever", "retriever_with_chat_history"]:
        event = _ndjson_event(StreamStatusEvent(data="Searching documents..."))

    elif chunk["event"] == "on_chain_end" and chunk["name"] in ["retriever", "retriever_with_chat_history"]:
        output = chunk["data"].get("output") or {}
        num_docs = len(output.get("documents", []))
        event = _ndjson_event(StreamStatusEvent(data=f"Found {num_docs} documents."))

    elif chunk["event"] == "on_chat_model_start":
        event = _ndjson_event(StreamStatusEvent(data="Generating answer..."))

    elif chunk["event"] == "on_chat_model_stream" and chunk["metadata"].get("langgraph_node") in [
        "response_synthesizer",
        "response_synthesizer_cohere",
    ]:
        content = chunk["data"]["chunk"].content
        if content:
            event = _ndjson_event(StreamContentEvent(data=content))

    elif chunk["name"] == "LangGraph" and chunk["event"] == "on_chain_end":
        output = chunk["data"].get("output") or {}
        if "documents" in output:
            citations = [CitationDocument(document=[doc.page_content], metadata=[doc.metadata]) for doc in output["documents"]]
            event = _ndjson_event(StreamCitationEvent(data=citations))

    return event


@router.post("/", tags=["rag"])
async def question_answer(rag: RAGRequest) -> QAResponse:
    """Answering the Question."""
    messages = [dict(m) for m in rag.messages]
    chain_result = await graph.with_config({"metadata": {"collection_name": rag.collection_name}}).ainvoke({"messages": messages})

    documents = [{"document": [doc.page_content], "metadata": [doc.metadata]} for doc in chain_result["documents"]]
    return QAResponse(answer=chain_result["messages"][-1].content, meta_data=documents)


@router.post(
    "/stream",
    tags=["rag"],
    response_class=NDJSONStreamingResponse,
    responses={
        200: {
            "description": "NDJSON event stream. Each line is a JSON object with a `type` and `data` field.",
            "content": {
                "application/x-ndjson": {
                    "schema": STREAM_EVENT_SCHEMA,
                    "examples": STREAM_EVENT_EXAMPLES,
                }
            },
        }
    },
)
async def question_answer_stream(rag: RAGRequest) -> AsyncIterable[str]:
    """Stream the RAG answering process as NDJSON events.

    Event types: status, content, citation, error.
    """
    messages = [dict(m) for m in rag.messages]

    yield _ndjson_event(StreamStatusEvent(data="Starting request..."))

    try:
        async for chunk in graph.with_config({"metadata": {"collection_name": rag.collection_name}}).astream_events({"messages": messages}, version="v2"):
            event = _stream_event_from_chunk(chunk)
            if event is not None:
                yield event

    except Exception:
        logger.exception("Error during RAG streaming")
        yield _ndjson_event(StreamErrorEvent(data="An internal error occurred during streaming."))
    else:
        yield _ndjson_event(StreamStatusEvent(data="Done."))
