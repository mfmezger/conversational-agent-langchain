"""Script that contains the Pydantic Models for internal Data handling."""

from pydantic import BaseModel


class RetrievalResults(BaseModel):
    """The Retrieval Results Model."""

    document: str
    metadata: dict
    score: float
