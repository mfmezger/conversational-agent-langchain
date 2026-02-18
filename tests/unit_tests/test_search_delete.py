from __future__ import annotations

from unittest.mock import MagicMock, patch

from inline_snapshot import snapshot
from qdrant_client.http.models.models import UpdateResult

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


@patch("agent.routes.delete.load_vec_db_conn")
def test_delete_vector(mock_load_conn, client) -> None:
    mock_client = MagicMock()
    mock_result = UpdateResult(operation_id=0, status="completed")
    mock_client.delete.return_value = mock_result
    mock_load_conn.return_value = mock_client

    response = client.delete("/embeddings/delete/test.pdf?collection_name=test_coll")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    mock_client.delete.assert_called_once()


@patch("agent.utils.retriever.qdrant_client")
@patch("agent.utils.retriever.sparse_embeddings")
@patch("agent.utils.retriever.QdrantVectorStore")
@patch("agent.utils.retriever.CohereEmbeddings")
def test_get_retriever(mock_embeddings, mock_vector_store, _mock_sparse, _mock_client) -> None:
    mock_vstore_instance = MagicMock()
    mock_vector_store.return_value = mock_vstore_instance

    from agent.utils.retriever import get_retriever

    get_retriever(k=5, collection_name="my_coll")

    mock_embeddings.assert_called_once()
    mock_vector_store.assert_called_once()
    mock_vstore_instance.as_retriever.assert_called_with(search_kwargs={"k": 5})
