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
    # Memory context identifiers for personalization
    user_id: str | None = Field(
        default=None,
        title="User ID",
        description="Unique user identifier for per-user memory persistence.",
    )
    session_id: str | None = Field(
        default=None,
        title="Session ID",
        description="Session identifier for session-scoped memory.",
    )
    agent_id: str | None = Field(
        default=None,
        title="Agent ID",
        description="Agent identifier for agent-scoped memory.",
    )


class EmbedTextRequest(BaseModel):
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
