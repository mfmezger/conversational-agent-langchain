"""OpenAI-compatible endpoints backed by the existing RAG graph."""

import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from loguru import logger
from starlette.types import Send

from agent.data_model.openai_compat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    OpenAIErrorResponse,
    ResponseFailure,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsesRequest,
    ResponsesResponse,
)
from agent.dependencies import GraphDep, VDBResourcesDep

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    from agent.utils.config import Config
    from agent.utils.vdb import VDBResources

_GENERATION_NODES = {"response_synthesizer", "response_synthesizer_cohere"}

router = APIRouter(prefix="/v1", tags=["OpenAI compatibility"])


async def _close_async_iterator(iterator: object) -> None:
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception as exc:
        logger.warning("Failed to close OpenAI-compatible stream iterator: {}", exc)


class _ClosingStreamingResponse(StreamingResponse):
    """Close the response iterator when ASGI sending stops early."""

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            await _close_async_iterator(self.body_iterator)


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


def _validate_model(model: str, config: "Config") -> None:
    if model != config.openai_compatible_model:
        raise OpenAIAPIError(
            status_code=404,
            message=f"The model `{model}` does not exist or is not available.",
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )


def _graph_config(resources: "VDBResources") -> "RunnableConfig":
    """Use application-owned resources and the server-configured collection."""
    return {
        "metadata": {"collection_name": resources.config.qdrant_collection_name},
        "configurable": {"vdb_resources": resources},
    }


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


async def _answer(messages: list[dict[str, str]], *, graph: "CompiledStateGraph", resources: "VDBResources") -> str:
    try:
        result = await graph.with_config(_graph_config(resources)).ainvoke({"messages": messages})
        return _answer_text(result)
    except Exception as exc:
        raise OpenAIAPIError(
            status_code=500,
            message="Internal server error.",
            error_type="server_error",
            param=None,
            code="internal_error",
        ) from exc


def _event_text_delta(event: dict[str, Any]) -> str | None:
    if event.get("event") != "on_chat_model_stream" or event.get("metadata", {}).get("langgraph_node") not in _GENERATION_NODES:
        return None
    content = event["data"]["chunk"].content
    if not content:
        return None
    if not isinstance(content, str):
        msg = "The RAG graph returned a non-text stream chunk."
        raise TypeError(msg)
    return content


async def _text_deltas(messages: list[dict[str, str]], *, graph: "CompiledStateGraph", resources: "VDBResources") -> AsyncIterator[str]:
    events = graph.with_config(_graph_config(resources)).astream_events({"messages": messages}, version="v2")
    failure_in_flight = False
    try:
        async for event in events:
            content = _event_text_delta(event)
            if content is not None:
                yield content
    except BaseException:
        failure_in_flight = True
        raise
    finally:
        close = getattr(events, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception as exc:
                if not failure_in_flight:
                    raise
                logger.warning("Failed to close upstream RAG event stream: {}", exc)


def _sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _response_sse(payload: dict[str, Any]) -> str:
    return f"event: {payload['type']}\n{_sse_data(payload)}"


def _response_object(*, response_id: str, model: str, created_at: float, message: ResponseOutputMessage | None) -> ResponsesResponse:
    completed = message is not None
    return ResponsesResponse(
        id=response_id,
        object="response",
        created_at=created_at,
        status="completed" if completed else "in_progress",
        completed_at=time.time() if completed else None,
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata={},
        model=model,
        output=[message] if message else [],
        parallel_tool_calls=False,
        temperature=None,
        tool_choice="none",
        tools=[],
        top_p=None,
        usage=None,
    )


def _failed_response(*, response_id: str, model: str, created_at: float, message: ResponseOutputMessage) -> ResponsesResponse:
    return ResponsesResponse(
        id=response_id,
        object="response",
        created_at=created_at,
        status="failed",
        completed_at=None,
        error=ResponseFailure(code="server_error", message="Internal server error."),
        incomplete_details=None,
        instructions=None,
        metadata={},
        model=model,
        output=[message],
        parallel_tool_calls=False,
        temperature=None,
        tool_choice="none",
        tools=[],
        top_p=None,
        usage=None,
    )


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
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
async def create_chat_completion(request: ChatCompletionRequest, graph: GraphDep, resources: VDBResourcesDep) -> ChatCompletionResponse | StreamingResponse:
    """Create a text-only Chat Completion using the configured RAG collection."""
    _validate_model(request.model, resources.config)
    messages = _request_messages(request)
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())

    if not request.stream:
        answer = await _answer(messages, graph=graph, resources=resources)
        return ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(role="assistant", content=answer, refusal=None),
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
        )

    async def stream() -> AsyncIterator[str]:
        yield _sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "", "refusal": None},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
        )
        deltas = _text_deltas(messages, graph=graph, resources=resources)
        try:
            try:
                async for delta in deltas:
                    yield _sse_data(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta},
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            except Exception as exc:
                logger.error("OpenAI-compatible Chat Completion stream failed: {}", exc)
                yield _sse_data(
                    {
                        "error": {
                            "message": "Internal server error.",
                            "type": "server_error",
                            "param": None,
                            "code": "internal_error",
                        }
                    }
                )
                return
        finally:
            await _close_async_iterator(deltas)
        yield _sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}],
            }
        )
        yield "data: [DONE]\n\n"

    return _ClosingStreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/responses",
    response_model=ResponsesResponse,
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
async def create_response(request: ResponsesRequest, graph: GraphDep, resources: VDBResourcesDep) -> ResponsesResponse | StreamingResponse:
    """Create a text-only Responses API object using the configured RAG collection."""
    _validate_model(request.model, resources.config)
    messages = _response_messages(request)
    response_id = f"resp_{uuid4().hex}"
    message_id = f"msg_{uuid4().hex}"
    created_at = time.time()

    if not request.stream:
        answer = await _answer(messages, graph=graph, resources=resources)
        return _response_object(
            response_id=response_id,
            model=request.model,
            created_at=created_at,
            message=ResponseOutputMessage(
                id=message_id,
                type="message",
                status="completed",
                role="assistant",
                content=[ResponseOutputText(type="output_text", text=answer, annotations=[])],
            ),
        )

    async def stream() -> AsyncIterator[str]:
        sequence_number = 0
        initial_response = _response_object(
            response_id=response_id,
            model=request.model,
            created_at=created_at,
            message=None,
        ).model_dump()
        yield _response_sse({"type": "response.created", "response": initial_response, "sequence_number": sequence_number})
        sequence_number += 1
        yield _response_sse({"type": "response.in_progress", "response": initial_response, "sequence_number": sequence_number})
        sequence_number += 1

        in_progress_message = ResponseOutputMessage(
            id=message_id,
            type="message",
            status="in_progress",
            role="assistant",
            content=[],
        ).model_dump()
        yield _response_sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": in_progress_message,
                "sequence_number": sequence_number,
            }
        )
        sequence_number += 1
        empty_part = ResponseOutputText(type="output_text", text="", annotations=[]).model_dump()
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
        deltas = _text_deltas(messages, graph=graph, resources=resources)
        try:
            try:
                async for delta in deltas:
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
            except Exception as exc:
                logger.error("OpenAI-compatible Responses stream failed: {}", exc)
                yield _response_sse(
                    {
                        "type": "error",
                        "code": "server_error",
                        "message": "Internal server error.",
                        "param": None,
                        "sequence_number": sequence_number,
                    }
                )
                sequence_number += 1
                partial_text = "".join(text_parts)
                failed_message = ResponseOutputMessage(
                    id=message_id,
                    type="message",
                    status="incomplete",
                    role="assistant",
                    content=[ResponseOutputText(type="output_text", text=partial_text, annotations=[])],
                )
                failed_response = _failed_response(
                    response_id=response_id,
                    model=request.model,
                    created_at=created_at,
                    message=failed_message,
                ).model_dump()
                yield _response_sse(
                    {
                        "type": "response.failed",
                        "response": failed_response,
                        "sequence_number": sequence_number,
                    }
                )
                return
        finally:
            await _close_async_iterator(deltas)

        text = "".join(text_parts)
        completed_part = ResponseOutputText(type="output_text", text=text, annotations=[]).model_dump()
        completed_message = ResponseOutputMessage(
            id=message_id,
            type="message",
            status="completed",
            role="assistant",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
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
        ).model_dump()
        yield _response_sse({"type": "response.completed", "response": completed_response, "sequence_number": sequence_number})

    return _ClosingStreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
