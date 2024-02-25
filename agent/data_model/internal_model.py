"""Script that contains the Pydantic Models for internal Datahandeling."""
from pydantic import BaseModel


class RetrievalResults(BaseModel):
    """The Retrieval Results Model."""

    document: str
    metadata: dict
    score: float
