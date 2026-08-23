"""Schemas for the supported OpenAI-compatible API subset."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictOpenAIRequest(BaseModel):
    """Reject OpenAI request fields that this RAG API does not implement."""

    model_config = ConfigDict(extra="forbid")


class ChatCompletionMessageRequest(StrictOpenAIRequest):
    """A text-only user or assistant chat message."""

    role: Literal["user", "assistant"]
    content: str


class ChatCompletionRequest(StrictOpenAIRequest):
    """Supported request subset for Chat Completions."""

    model: str = Field(min_length=1, description="Configured public RAG model alias.")
    messages: list[ChatCompletionMessageRequest] = Field(min_length=1)
    stream: bool = False

    @model_validator(mode="after")
    def require_final_user_message(self) -> Self:
        """Require a question for the RAG pipeline to answer."""
        if self.messages[-1].role != "user":
            msg = "The final message must have role 'user'."
            raise ValueError(msg)
        return self


class ResponseInputMessage(StrictOpenAIRequest):
    """A text-only input message accepted by the Responses endpoint."""

    role: Literal["user", "assistant"]
    content: str


class ResponsesRequest(StrictOpenAIRequest):
    """Supported request subset for the Responses API."""

    model: str = Field(min_length=1, description="Configured public RAG model alias.")
    input: str | list[ResponseInputMessage]
    stream: bool = False

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        """Require non-empty input ending in a user message."""
        if isinstance(self.input, str):
            if not self.input:
                msg = "Input must not be empty."
                raise ValueError(msg)
        elif not self.input:
            msg = "Input messages must not be empty."
            raise ValueError(msg)
        elif self.input[-1].role != "user":
            msg = "The final input message must have role 'user'."
            raise ValueError(msg)
        return self


class OpenAIErrorDetail(BaseModel):
    """Error details used by the OpenAI-compatible routes."""

    message: str
    type: str
    param: str | None
    code: str


class OpenAIErrorResponse(BaseModel):
    """OpenAI-compatible error envelope."""

    error: OpenAIErrorDetail


class ChatCompletionResponseMessage(BaseModel):
    """Assistant message returned in a Chat Completion."""

    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """Single RAG answer choice."""

    index: Literal[0] = 0
    message: ChatCompletionResponseMessage
    finish_reason: Literal["stop"] = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible non-streaming Chat Completion."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class ResponseOutputText(BaseModel):
    """Text content in a Responses output message."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    """Assistant output item in a Responses object."""

    id: str
    type: Literal["message"] = "message"
    status: Literal["in_progress", "completed"]
    role: Literal["assistant"] = "assistant"
    content: list[ResponseOutputText]


class ResponsesResponse(BaseModel):
    """OpenAI-compatible non-streaming Responses object."""

    id: str
    object: Literal["response"] = "response"
    created_at: float
    status: Literal["in_progress", "completed"]
    completed_at: float | None = None
    error: None = None
    incomplete_details: None = None
    model: str
    output: list[ResponseOutputMessage]
    parallel_tool_calls: Literal[False] = False
    tool_choice: Literal["none"] = "none"
    tools: list[dict[str, Any]] = Field(default_factory=list)
