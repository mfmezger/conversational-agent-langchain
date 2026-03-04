"""State definitions for the agent."""

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """State of the Agent."""

    query: str
    documents: list[Document]
    messages: Annotated[list[BaseMessage], add_messages]
    retry_count: int


class Grade(BaseModel):
    """Binary score for relevance check."""

    is_relevant: bool = Field(description="True if the documents are relevant to the question, False otherwise")
