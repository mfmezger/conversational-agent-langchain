from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest

# Mock the vector db initialization to avoid connecting to real Qdrant during tests
with patch("agent.utils.vdb.initialize_all_vector_dbs"):
    from agent.api import app

client = TestClient(app)

def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the RAG Backend" in response.text

def test_docs_endpoint():
    """Test that the OpenAPI docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200

@patch("agent.api.search.router")
def test_search_endpoint_structure(mock_router):
    """
    Test that the search endpoint exists.
    Note: We are not testing full logic here to avoid mocking the entire graph/retriever
    which is complex due to the current architecture.
    """
    # Just verifying the route is registered
    routes = [route.path for route in app.routes]
    assert "/semantic/search" in routes
