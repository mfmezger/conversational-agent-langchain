"""Test configuration for all tests."""

from unittest.mock import MagicMock, patch


def pytest_configure(config):
    """Configure pytest with mock settings before any imports."""
    patch("agent.utils.vdb.QdrantClient").start()
    patch("agent.utils.vdb.sparse_embeddings").start()
    patch("agent.utils.vdb.QdrantVectorStore").start()
    patch("agent.utils.vdb.FastEmbedSparse").start()

    mock_client = MagicMock()
    patch("agent.utils.vdb.qdrant_client", mock_client).start()
    mock_client.collection_exists.return_value = False
    mock_client.get_fastembed_sparse_vector_params.return_value = {"sparse_vectors_config": {"fast-sparse-bm25": {"index": {"type": "sparse"}}}}
    mock_client.set_sparse_model.return_value = None
