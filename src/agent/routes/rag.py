"""The RAG Routes."""

import logging
from collections.abc import AsyncIterable

from fastapi import APIRouter

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


StreamEvent = StreamStatusEvent | StreamContentEvent | StreamCitationEvent | StreamErrorEvent

STREAM_EVENT_ITEM_SCHEMA = {
    "anyOf": [
        StreamStatusEvent.model_json_schema(),
        StreamContentEvent.model_json_schema(),
        StreamCitationEvent.model_json_schema(),
        StreamErrorEvent.model_json_schema(),
    ]
}


def _stream_event_from_chunk(chunk: dict) -> StreamEvent | None:
    """Map a LangGraph chunk to one JSON Lines event."""
    event = None
    chunk_event = chunk.get("event")
    chunk_name = chunk.get("name")
    chunk_data = chunk.get("data") or {}
    chunk_metadata = chunk.get("metadata") or {}

    if chunk_event == "on_chain_start" and chunk_name in ["retriever", "retriever_with_chat_history"]:
        event = StreamStatusEvent(data="Searching documents...")

    elif chunk_event == "on_chain_end" and chunk_name in ["retriever", "retriever_with_chat_history"]:
        output = chunk_data.get("output") or {}
        num_docs = len(output.get("documents") or [])
        event = StreamStatusEvent(data=f"Found {num_docs} documents.")

    elif chunk_event == "on_chat_model_start":
        event = StreamStatusEvent(data="Generating answer...")

    elif chunk_event == "on_chat_model_stream" and chunk_metadata.get("langgraph_node") in [
        "response_synthesizer",
        "response_synthesizer_cohere",
    ]:
        content = getattr(chunk_data.get("chunk"), "content", None)
        if content:
            event = StreamContentEvent(data=content)

    elif chunk_name == "LangGraph" and chunk_event == "on_chain_end":
        output = chunk_data.get("output") or {}
        if "documents" in output:
            citations = [CitationDocument(document=[doc.page_content], metadata=[doc.metadata]) for doc in output.get("documents") or []]
            event = StreamCitationEvent(data=citations)

    return event


@router.post("/", tags=["rag"])
async def question_answer(rag: RAGRequest) -> QAResponse:
    """Answering the Question."""
    messages = [dict(m) for m in (rag.messages or [])]
    chain_result = await graph.with_config({"metadata": {"collection_name": rag.collection_name}}).ainvoke({"messages": messages})

    documents = [{"document": [doc.page_content], "metadata": [doc.metadata]} for doc in chain_result.get("documents", [])]
    messages_out = chain_result.get("messages", [])
    answer = messages_out[-1].content if messages_out else ""
    return QAResponse(answer=answer, meta_data=documents)


@router.post(
    "/stream",
    tags=["rag"],
    responses={
        200: {
            "description": "JSON Lines event stream. Each line is a JSON object with a `type` and `data` field.",
            "content": {
                "application/jsonl": {
                    "itemSchema": STREAM_EVENT_ITEM_SCHEMA,
                }
            },
        }
    },
)
async def question_answer_stream(rag: RAGRequest) -> AsyncIterable[StreamEvent]:
    """Stream the RAG answering process as JSON Lines events.

    Event types: status, content, citation, error.
    """
    messages = [dict(m) for m in (rag.messages or [])]

    yield StreamStatusEvent(data="Starting request...")

    try:
        async for chunk in graph.with_config({"metadata": {"collection_name": rag.collection_name}}).astream_events({"messages": messages}, version="v2"):
            event = _stream_event_from_chunk(chunk)
            if event is not None:
                yield event

    except Exception:
        logger.exception("Error during RAG streaming")
        yield StreamErrorEvent(data="An internal error occurred during streaming.")
    else:
        yield StreamStatusEvent(data="Done.")
