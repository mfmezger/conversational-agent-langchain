"""Script that contains the Pydantic Models for the Rest API Response."""

from enum import Enum

from pydantic import BaseModel, Field


class Status(str, Enum):
    """Status."""

    SUCCESS = "success"
    FAILURE = "failure"


class SearchResponse(BaseModel):
    """The request parameters for explaining the output."""

    text: str = Field(..., title="Text", description="The text of the document.")
    page: int = Field(..., title="Page", description="The page of the document.")
    source: str = Field(..., title="Source", description="The source of the document.")


class EmbeddingResponse(BaseModel):
    """The Response for the Embedding endpoint."""

    status: Status = Field(Status.SUCCESS, title="Status", description="The status of the request.")
    files: list[str] = Field([], title="Files", description="The list of files that were embedded.")


class QAResponse(BaseModel):
    """The Response for the QA endpoint."""

    answer: str = Field(..., title="Answer", description="The answer to the question.")
    meta_data: list


class CitationDocument(BaseModel):
    """A streamed citation payload entry."""

    document: list[str] = Field(..., title="Document", description="The retrieved document content.")
    metadata: list[dict] = Field(..., title="Metadata", description="Metadata for the retrieved document.")


class StreamStatusEvent(BaseModel):
    """A streamed status update."""

    type: str = Field(default="status", title="Type", description="The event type.")
    data: str = Field(..., title="Data", description="The status message.")


class StreamContentEvent(BaseModel):
    """A streamed answer content chunk."""

    type: str = Field(default="content", title="Type", description="The event type.")
    data: str = Field(..., title="Data", description="A streamed text chunk.")


class StreamCitationEvent(BaseModel):
    """A streamed citation event."""

    type: str = Field(default="citation", title="Type", description="The event type.")
    data: list[CitationDocument] = Field(..., title="Data", description="The retrieved citation payload.")


class StreamErrorEvent(BaseModel):
    """A streamed error event."""

    type: str = Field(default="error", title="Type", description="The event type.")
    data: str = Field(..., title="Data", description="The error message.")


class ExplainQAResponse(BaseModel):
    """The Response for the Explain QA endpoint."""

    answer: str = Field(..., title="Answer", description="The answer to the question.")
    meta_data: list
    explanation: str = Field(..., title="Explanation", description="The explanation for the answer.")
    text: str = Field(..., title="Text", description="The text of the document.")
    score: float = Field(..., title="Score", description="The score of the document.")
