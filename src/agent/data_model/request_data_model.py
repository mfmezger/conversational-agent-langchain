"""Script that contains the Pydantic Models for the Rest Request."""

from pydantic import BaseModel, Field


class SearchParams(BaseModel):
    """The request parameters for searching the database."""

    query: str = Field(..., title="Query", description="The search query.")
    k: int = Field(3, title="Amount", description="The number of search results to return.")
    collection_name: str = Field(
        default="default",
        title="Collection Name",
        description="The name of the collection to search in.",
    )


class ChatMessages(BaseModel):
    """The Chat Messages Model."""

    role: str = Field(
        ...,
        title="Role",
        description="The role of the sender can be either user or assistant.",
    )
    content: str = Field(default=..., title="Content", description="The content of the message.")


class OpenAIChatRequest(BaseModel):
    """Request for the OpenAI compatible chat completions endpoint."""

    model: str = Field(
        default="default",
        title="Model",
        description="The collection name to use for retrieval. Must be alphanumeric with hyphens/underscores.",
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$",
    )
    messages: list[ChatMessages] = Field(
        ...,
        title="Messages",
        description="A list of messages comprising the conversation so far.",
    )
    stream: bool | None = Field(
        default=False,
        title="Stream",
        description="If set, partial message deltas will be sent, like in ChatGPT.",
    )


class RAGRequest(BaseModel):
    """Request for the QA endpoint."""

    messages: list[ChatMessages] | None = Field(
        default=[
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
        ],
        title="History",
        description="A list of previous questions and answers to include in the context.",
    )
    collection_name: str = Field(
        default="default",
        title="Collection Name",
        description="The name of the collection to search in.",
    )


class EmbeddTextRequest(BaseModel):
    """The request parameters for embedding text."""

    text: str = Field(..., title="Text", description="The text to embed.")
    file_name: str = Field(
        ...,
        title="File Name",
        description="The name of the file to save the embedded text to.",
    )
    separator: str = Field(
        "###",
        title="Separator",
        description="The separator to use between embedded texts.",
    )
