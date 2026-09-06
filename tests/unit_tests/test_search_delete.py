from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from inline_snapshot import snapshot
from qdrant_client.http.models.models import UpdateResult

from agent.utils.config import Config
from agent.utils.retriever import get_retriever
from agent.utils.vdb import VDBResources
from tests.fakes.rag import FakeAsyncRetriever, FakeDoc


@patch("agent.routes.search.get_retriever")
def test_search_documents_found(mock_get_retriever, client) -> None:
    mock_get_retriever.return_value = FakeAsyncRetriever([FakeDoc(page_content="content", metadata={"page": 1, "source": "test.pdf"})])

    response = client.post("/semantic/search", json={"query": "test", "collection_name": "test_coll", "k": 2})

    assert response.status_code == 200
    assert response.json() == snapshot([{"text": "content", "page": 1, "source": "test.pdf"}])


@patch("agent.routes.search.get_retriever")
def test_search_no_documents(mock_get_retriever, client) -> None:
    mock_get_retriever.return_value = FakeAsyncRetriever([])

    response = client.post("/semantic/search", json={"query": "test", "collection_name": "test_coll"})

    assert response.status_code == 200
    assert response.json() == snapshot({"message": "No documents found."})


@patch("agent.routes.delete.delete_documents_by_source", new_callable=AsyncMock)
def test_delete_vector(mock_delete, client) -> None:
    mock_result = UpdateResult(operation_id=0, status="completed")
    mock_delete.return_value = mock_result

    response = client.delete("/embeddings/delete/test.pdf?collection_name=test_coll")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    mock_delete.assert_awaited_once_with(
        client=client.app.state.vdb_resources.async_client,
        collection_name="test_coll",
        source="test.pdf",
    )


@patch("agent.utils.retriever.get_vector_store")
@patch("agent.utils.retriever.get_embedding_model")
def test_get_retriever(mock_get_embedding_model, mock_get_vector_store) -> None:
    mock_vstore_instance = MagicMock()
    mock_get_vector_store.return_value = mock_vstore_instance

    resources = VDBResources(
        config=Config(),
        sync_client=MagicMock(),
        async_client=MagicMock(),
        sparse_embeddings=MagicMock(),
    )
    mock_embedding = MagicMock()
    mock_get_embedding_model.return_value = mock_embedding

    get_retriever(resources=resources, k=5, collection_name="my_coll")

    mock_get_embedding_model.assert_called_once_with(resources.config)
    mock_get_vector_store.assert_called_once_with(
        resources=resources,
        collection_name="my_coll",
        embedding=mock_embedding,
    )
    mock_vstore_instance.as_retriever.assert_called_with(search_kwargs={"k": 5})
