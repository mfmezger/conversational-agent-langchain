"""Test the aleph alpha service."""
import os
import uuid
from pathlib import Path

import pytest

from agent.backend.aleph_alpha_service import AlephAlphaService
from agent.data_model.request_data_model import (
    Filtering,
    RAGRequest,
    SearchRequest,
)


# for every test load the service with the token
@pytest.fixture()
def service() -> AlephAlphaService:
    """Prepare the service."""
    token = os.getenv("ALEPH_ALPHA_API_KEY")
    assert token is not None
    return AlephAlphaService(collection_name="aleph_alpha", token=token)


def test_create_collection(service: AlephAlphaService) -> None:
    """Test the create_collection function."""
    response = service.create_collection(str(uuid.uuid4()))
    assert response is True


def test_summarize_text(service: AlephAlphaService) -> None:
    """Test the summarize_text function."""
    with Path("tests/resources/albert.txt").open() as f:
        text = f.read()
    response = service.summarize_text(text)
    assert response is not None


def test_generation(service: AlephAlphaService) -> None:
    """Test the generation function."""
    response = service.generate("What is the meaning of life?")
    assert response is not None


def test_embed_documents(service: AlephAlphaService) -> None:
    """Test the embedd_documents function."""
    response = service.embed_documents(folder="tests/resources")
    assert response is not None


def test_search(service: AlephAlphaService) -> None:
    """Test the search function."""
    response = service.search(SearchRequest(query="Was ist Attention?", amount=3), Filtering(threshold=0.0, collection_name="aleph_alpha"))
    assert response is not None
    assert len(response) > 0


def test_rag(service: AlephAlphaService) -> None:
    """Test the rag function."""
    response = service.rag(
        RAGRequest(language="detect", history={}),
        SearchRequest(
            query="Was ist Attention?",
            amount=3,
        ),
        Filtering(threshold=0.0, collection_name="aleph_alpha"),
    )
    assert response is not None
    assert len(response) > 0
