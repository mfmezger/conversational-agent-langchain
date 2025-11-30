from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
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

@patch("agent.routes.search.get_retriever")
def test_search_endpoint_execution(mock_get_retriever):
    """
    Test that the search endpoint executes the graph logic.
    We mock the retriever to avoid hitting Qdrant.
    """
    # Mock the retriever instance and its invoke method
    mock_retriever_instance = MagicMock()
    # Mock ainvoke for async call
    mock_retriever_instance.ainvoke = AsyncMock(return_value=[
        MagicMock(page_content="Test document content", metadata={"source": "test.pdf", "page": 1})
    ])
    mock_get_retriever.return_value = mock_retriever_instance

    # Test the search endpoint
    response = client.post(
        "/semantic/search",
        json={"query": "test", "collection_name": "default", "k": 4}
    )
    assert response.status_code == 200
    data = response.json()
    # Depending on how search.py is implemented, it might return a list of docs or a dict
    # Let's assume it returns a list of docs based on previous context
    assert len(data) > 0
    assert data[0]["text"] == "Test document content"
