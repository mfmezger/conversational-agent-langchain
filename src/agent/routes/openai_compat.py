"""OpenAI-compatible endpoints backed by the existing RAG graph."""

import json
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agent.data_model.openai_compat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    OpenAIErrorResponse,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsesRequest,
    ResponsesResponse,
)
from agent.routes.rag import graph
from agent.utils.config import Config

_GENERATION_NODES = {"response_synthesizer", "response_synthesizer_cohere"}

settings = Config()
router = APIRouter(prefix="/v1", tags=["OpenAI compatibility"])


class OpenAIAPIError(Exception):
    """An error rendered with an OpenAI-compatible envelope."""

    def __init__(self, *, status_code: int, message: str, error_type: str, param: str | None, code: str) -> None:
        """Initialize a public API error."""
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code


def _validate_model(model: str) -> None:
    if model != settings.openai_compatible_model:
        raise OpenAIAPIError(
            status_code=404,
            message=f"The model `{model}` does not exist or is not available.",
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )


def _graph_config() -> dict[str, dict[str, str]]:
    """Use the server-configured collection, independently of the public model alias."""
    return {"metadata": {"collection_name": settings.qdrant_collection_name}}


def _answer_text(chain_result: dict[str, Any]) -> str:
    content = chain_result["messages"][-1].content
    if not isinstance(content, str):
        msg = "The RAG graph returned non-text output."
        raise TypeError(msg)
    return content


def _request_messages(request: ChatCompletionRequest) -> list[dict[str, str]]:
    return [message.model_dump() for message in request.messages]


def _response_messages(request: ResponsesRequest) -> list[dict[str, str]]:
    if isinstance(request.input, str):
        return [{"role": "user", "content": request.input}]
    return [message.model_dump() for message in request.input]


async def _answer(messages: list[dict[str, str]]) -> str:
    try:
        result = await graph.with_config(_graph_config()).ainvoke({"messages": messages})
        return _answer_text(result)
    except Exception as exc:
        raise OpenAIAPIError(
            status_code=500,
            message="Internal server error.",
            error_type="server_error",
            param=None,
            code="internal_error",
        ) from exc


async def _text_deltas(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    events = graph.with_config(_graph_config()).astream_events({"messages": messages}, version="v2")
    async for event in events:
        if event.get("event") != "on_chat_model_stream" or event.get("metadata", {}).get("langgraph_node") not in _GENERATION_NODES:
            continue
        content = event["data"]["chunk"].content
        if not content:
            continue
        if not isinstance(content, str):
            msg = "The RAG graph returned a non-text stream chunk."
            raise TypeError(msg)
        yield content


def _sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _response_sse(payload: dict[str, Any]) -> str:
    return f"event: {payload['type']}\n{_sse_data(payload)}"


def _response_object(*, response_id: str, model: str, created_at: float, message: ResponseOutputMessage | None) -> ResponsesResponse:
    completed = message is not None
    return ResponsesResponse(
        id=response_id,
        created_at=created_at,
        status="completed" if completed else "in_progress",
        completed_at=time.time() if completed else None,
        model=model,
        output=[message] if message else [],
    )


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    response_model_exclude_none=True,
    summary="Create a RAG-backed chat completion",
    description="Supports text-only user/assistant messages, one configured model alias, and optional SSE streaming.",
    responses={
        200: {
            "description": "Chat Completion JSON or SSE stream",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
        400: {"model": OpenAIErrorResponse, "description": "Invalid or unsupported request"},
        404: {"model": OpenAIErrorResponse, "description": "Unknown model alias"},
        500: {"model": OpenAIErrorResponse, "description": "RAG pipeline failure"},
    },
)
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse | StreamingResponse:
    """Create a text-only Chat Completion using the configured RAG collection."""
    _validate_model(request.model)
    messages = _request_messages(request)
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())

    if not request.stream:
        answer = await _answer(messages)
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChoice(message=ChatCompletionResponseMessage(content=answer))],
        )

    async def stream() -> AsyncIterator[str]:
        yield _sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
        )
        async for delta in _text_deltas(messages):
            yield _sse_data(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
            )
        yield _sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/responses",
    response_model=ResponsesResponse,
    response_model_exclude_none=True,
    summary="Create a RAG-backed response",
    description="Supports string input or text-only user/assistant input messages, one configured model alias, and optional SSE streaming.",
    responses={
        200: {
            "description": "Responses object JSON or SSE stream",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
        400: {"model": OpenAIErrorResponse, "description": "Invalid or unsupported request"},
        404: {"model": OpenAIErrorResponse, "description": "Unknown model alias"},
        500: {"model": OpenAIErrorResponse, "description": "RAG pipeline failure"},
    },
)
async def create_response(request: ResponsesRequest) -> ResponsesResponse | StreamingResponse:
    """Create a text-only Responses API object using the configured RAG collection."""
    _validate_model(request.model)
    messages = _response_messages(request)
    response_id = f"resp_{uuid4().hex}"
    message_id = f"msg_{uuid4().hex}"
    created_at = time.time()

    if not request.stream:
        answer = await _answer(messages)
        return _response_object(
            response_id=response_id,
            model=request.model,
            created_at=created_at,
            message=ResponseOutputMessage(
                id=message_id,
                status="completed",
                content=[ResponseOutputText(text=answer)],
            ),
        )

    async def stream() -> AsyncIterator[str]:
        sequence_number = 0
        initial_response = _response_object(
            response_id=response_id,
            model=request.model,
            created_at=created_at,
            message=None,
        ).model_dump(exclude_none=True)
        yield _response_sse({"type": "response.created", "response": initial_response, "sequence_number": sequence_number})
        sequence_number += 1
        yield _response_sse({"type": "response.in_progress", "response": initial_response, "sequence_number": sequence_number})
        sequence_number += 1

        in_progress_message = ResponseOutputMessage(id=message_id, status="in_progress", content=[]).model_dump()
        yield _response_sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": in_progress_message,
                "sequence_number": sequence_number,
            }
        )
        sequence_number += 1
        empty_part = ResponseOutputText(text="").model_dump()
        yield _response_sse(
            {
                "type": "response.content_part.added",
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "part": empty_part,
                "sequence_number": sequence_number,
            }
        )
        sequence_number += 1

        text_parts: list[str] = []
        async for delta in _text_deltas(messages):
            text_parts.append(delta)
            yield _response_sse(
                {
                    "type": "response.output_text.delta",
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta,
                    "logprobs": [],
                    "sequence_number": sequence_number,
                }
            )
            sequence_number += 1

        text = "".join(text_parts)
        completed_part = ResponseOutputText(text=text).model_dump()
        completed_message = ResponseOutputMessage(
            id=message_id,
            status="completed",
            content=[ResponseOutputText(text=text)],
        )
        yield _response_sse(
            {
                "type": "response.output_text.done",
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
                "logprobs": [],
                "sequence_number": sequence_number,
            }
        )
        sequence_number += 1
        yield _response_sse(
            {
                "type": "response.content_part.done",
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "part": completed_part,
                "sequence_number": sequence_number,
            }
        )
        sequence_number += 1
        yield _response_sse(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": completed_message.model_dump(),
                "sequence_number": sequence_number,
            }
        )
        sequence_number += 1
        completed_response = _response_object(
            response_id=response_id,
            model=request.model,
            created_at=created_at,
            message=completed_message,
        ).model_dump(exclude_none=True)
        yield _response_sse({"type": "response.completed", "response": completed_response, "sequence_number": sequence_number})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
