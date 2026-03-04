import pytest
from unittest.mock import MagicMock, patch
from agent.utils.vdb import initialize_vector_db, generate_collection, init_vdb, initialize_all_vector_dbs
from agent.utils.config import Config

@patch("agent.utils.vdb.load_vec_db_conn")
def test_initialize_vector_db_exists(mock_load_conn):
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_load_conn.return_value = mock_client

    initialize_vector_db("test_coll", 1536)

    mock_client.collection_exists.assert_called_with(collection_name="test_coll")
    # Should NOT call create_collection
    mock_client.create_collection.assert_not_called()

@patch("agent.utils.vdb.load_vec_db_conn")
def test_initialize_vector_db_not_exists(mock_load_conn):
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    mock_load_conn.return_value = mock_client

    initialize_vector_db("test_coll", 1536)

    mock_client.collection_exists.assert_called_with(collection_name="test_coll")
    mock_client.create_collection.assert_called_once()
    mock_client.set_sparse_model.assert_called_with(embedding_model_name="Qdrant/bm25")

@patch("agent.utils.vdb.QdrantVectorStore")
@patch("agent.utils.vdb.FastEmbedSparse")
def test_init_vdb(mock_sparse, mock_vstore):
    mock_embedding = MagicMock()

    init_vdb("test_coll", mock_embedding)

    mock_vstore.assert_called_once()
    args, kwargs = mock_vstore.call_args
    assert kwargs["collection_name"] == "test_coll"
    assert kwargs["embedding"] == mock_embedding

@patch("agent.utils.vdb.initialize_vector_db")
def test_initialize_all_vector_dbs(mock_init_vdb):
    config = Config()
    # We can mock config values if needed, but default is fine
    initialize_all_vector_dbs(config)

    mock_init_vdb.assert_called_once()
