"""API tests."""

from __future__ import annotations

from http import HTTPStatus
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.integration


def test_read_root(client) -> None:
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK


@patch("agent.routes.collection.initialize_vector_db", return_value=None)
def test_create_collection(_mock_init_db, client) -> None:
    collection_name = "test_collection"
    response = client.post(f"/collection/create/{collection_name}", params={"embeddings_size": 1536})

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"message": f"Collection {collection_name} created."}
