"""Test the aleph alpha service."""
import os
import uuid
from pathlib import Path

from agent.backend.aleph_alpha_service import AlephAlphaService


# for every test load the service with the token
def prepare_service() -> AlephAlphaService:
    """Prepare the service."""
    token = os.getenv("ALEPH_ALPHA_API_KEY")
    assert token is not None
    return AlephAlphaService(token)


def test_create_collection() -> None:
    """Test the create_collection function."""
    service = prepare_service()
    response = service.create_collection(str(uuid.uuid4()))
    assert response is not None
    assert response["message"] == "Collection invoice created with embeddings size 512."


def test_summarize_text() -> None:
    """Test the summarize_text function."""
    service = prepare_service()

    with Path("tests/resources/albert.txt").open() as f:
        text = f.read()
    response = service.summarize_text(text)
    assert response is not None


def test_generation() -> None:
    """Test the generation function."""
    service = prepare_service()
    response = service.generate("What is the meaning of life?")
    assert response is not None


def test_embedd_documents() -> None:
    """Test the embedd_documents function."""
    service = prepare_service()
    response = service.embedd_documents(folder="tests/resources")
    assert response is not None


def test_search() -> None:
    """Test the search function."""
    service = prepare_service()
    response = service.search("What is the meaning of life?")
    assert response is not None
    assert len(response) > 0


def test_rag() -> None:
    """Test the rag function."""
    service = prepare_service()
    response = service.rag("What is the meaning of life?")
    assert response is not None
    assert len(response) > 0
