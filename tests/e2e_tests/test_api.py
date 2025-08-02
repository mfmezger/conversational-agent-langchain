"""API Tests."""
from http import HTTPStatus
from pathlib import Path
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from agent.api import app

@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Yield a test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def resources_path() -> Path:
    """Return the path to the resources folder."""
    return Path("tests/resources")


def test_read_root(client: TestClient) -> None:
    """Test the root method."""
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK


@pytest.mark.parametrize("provider", ["cohere", "ollama"])
def test_create_collection(client: TestClient, provider: str) -> None:
    """Test the create_collection function."""
    collection_name = "test_collection"
    response = client.post(
        f"/collection/create/{collection_name}",
        params={"embeddings_size": 1536},
    )
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"message": f"Collection {collection_name} created."}
