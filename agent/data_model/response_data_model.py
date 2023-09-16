"""Script that contains the Pydantic Models for the Rest Response."""
from typing import List

from pydantic import BaseModel, Field


class SearchResponse(BaseModel):
    """The request parameters for explaining the output."""

    text: str = Field(..., title="Text", description="The text of the document.")
    page: int = Field(..., title="Page", description="The page of the document.")
    source: str = Field(..., title="Source", description="The source of the document.")
    score: float = Field(..., title="Score", description="The score of the document.")


class EmbeddingResponse(BaseModel):
    """The Response for the Embedding endpoint."""

    status: str = Field(..., title="Status", description="The status of the request.")
    files: List[str] = Field(..., title="Files", description="The list of files that were embedded.")
