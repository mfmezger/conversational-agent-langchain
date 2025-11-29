import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from agent.api import app
from agent.utils.retriever import get_retriever
from qdrant_client.http.models.models import UpdateResult

client = TestClient(app)

# --- Tests for Search Route ---

@pytest.mark.asyncio
@patch("agent.routes.search.get_retriever")
async def test_search_documents_found(mock_get_retriever):
    # Mock retriever
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "content"
    mock_doc.metadata = {"page": 1, "source": "test.pdf"}
    mock_retriever.ainvoke = AsyncMock(return_value=[mock_doc])
    mock_get_retriever.return_value = mock_retriever

    payload = {
        "query": "test",
        "collection_name": "test_coll",
        "k": 2
    }

    response = client.post("/semantic/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["text"] == "content"
    assert data[0]["page"] == 1
    assert data[0]["source"] == "test.pdf"

@pytest.mark.asyncio
@patch("agent.routes.search.get_retriever")
async def test_search_no_documents(mock_get_retriever):
    # Mock retriever returning empty list
    mock_retriever = MagicMock()
    mock_retriever.ainvoke = AsyncMock(return_value=[])
    mock_get_retriever.return_value = mock_retriever

    payload = {
        "query": "test",
        "collection_name": "test_coll"
    }

    response = client.post("/semantic/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "No documents found."

# --- Tests for Delete Route ---

@pytest.mark.asyncio
@patch("agent.routes.delete.load_vec_db_conn")
async def test_delete_vector(mock_load_conn):
    mock_client = MagicMock()
    # Mock delete result
    mock_result = UpdateResult(operation_id=0, status="completed")
    mock_client.delete.return_value = mock_result
    mock_load_conn.return_value = mock_client

    response = client.delete("/embeddings/delete/test.pdf?collection_name=test_coll")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    mock_client.delete.assert_called_once()

# --- Tests for Retriever Utils ---

@patch("agent.utils.retriever.QdrantClient")
@patch("agent.utils.retriever.QdrantVectorStore")
@patch("agent.utils.retriever.CohereEmbeddings")
def test_get_retriever(mock_embeddings, mock_vector_store, mock_client):
    mock_vstore_instance = MagicMock()
    mock_vector_store.return_value = mock_vstore_instance

    retriever = get_retriever(k=5, collection_name="my_coll")

    mock_embeddings.assert_called_once()
    mock_client.assert_called_once()
    mock_vector_store.assert_called_once()
    mock_vstore_instance.as_retriever.assert_called_with(search_kwargs={"k": 5})
