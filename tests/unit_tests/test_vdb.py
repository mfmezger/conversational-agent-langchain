import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.utils.config import Config
from agent.utils.vdb import (
    VDBResources,
    create_vdb_resources,
    delete_documents_by_source,
    get_vector_store,
    init_vdb,
    initialize_all_vector_dbs,
    initialize_vector_db,
)


def test_import_does_not_construct_resources() -> None:
    code = """
from unittest.mock import patch
with (
    patch('agent.utils.config.Config', side_effect=AssertionError('Config constructed')),
    patch('qdrant_client.QdrantClient', side_effect=AssertionError('sync client constructed')),
    patch('qdrant_client.AsyncQdrantClient', side_effect=AssertionError('async client constructed')),
    patch('langchain_qdrant.FastEmbedSparse', side_effect=AssertionError('sparse embeddings constructed')),
):
    import agent.utils.vdb
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def make_resources() -> VDBResources:
    async_client = MagicMock()
    async_client.close = AsyncMock()
    return VDBResources(
        config=Config(),
        sync_client=MagicMock(),
        async_client=async_client,
        sparse_embeddings=MagicMock(),
    )


@patch("agent.utils.vdb.FastEmbedSparse")
@patch("agent.utils.vdb.AsyncQdrantClient")
@patch("agent.utils.vdb.QdrantClient")
def test_create_vdb_resources_constructs_owned_resources(sync_client_cls, async_client_cls, sparse_cls):
    config = Config(qdrant_url="http://qdrant", qdrant_port=6334)

    resources = create_vdb_resources(config)

    assert resources.config is config
    assert resources.sync_client is sync_client_cls.return_value
    assert resources.async_client is async_client_cls.return_value
    assert resources.sparse_embeddings is sparse_cls.return_value
    sync_client_cls.assert_called_once_with(
        location="http://qdrant",
        port=6334,
        api_key=config.qdrant_api_key,
        prefer_grpc=config.qdrant_prefer_grpc,
    )
    async_client_cls.assert_called_once_with(
        location="http://qdrant",
        port=6334,
        api_key=config.qdrant_api_key,
        prefer_grpc=config.qdrant_prefer_grpc,
    )


def test_initialize_vector_db_exists():
    client = MagicMock()
    client.collection_exists.return_value = True

    initialize_vector_db(client, "test_coll", 1536)

    client.collection_exists.assert_called_with(collection_name="test_coll")
    client.create_collection.assert_not_called()


def test_initialize_vector_db_not_exists():
    client = MagicMock()
    client.collection_exists.return_value = False

    initialize_vector_db(client, "test_coll", 1536)

    client.collection_exists.assert_called_with(collection_name="test_coll")
    client.create_collection.assert_called_once()
    client.set_sparse_model.assert_called_with(embedding_model_name="Qdrant/bm25")


@patch("agent.utils.vdb.QdrantVectorStore")
def test_init_vdb_uses_sync_client(mock_vstore):
    resources = make_resources()
    embedding = MagicMock()

    init_vdb(resources, "test_coll", embedding)

    mock_vstore.assert_called_once()
    kwargs = mock_vstore.call_args.kwargs
    assert kwargs["client"] is resources.sync_client
    assert kwargs["collection_name"] == "test_coll"
    assert kwargs["embedding"] is embedding
    assert kwargs["sparse_embedding"] is resources.sparse_embeddings


def test_get_vector_store_is_cached_per_resource():
    resources = make_resources()
    with patch("agent.utils.vdb.init_vdb") as mock_init_vdb:
        store = MagicMock()
        mock_init_vdb.return_value = store
        embedding = MagicMock()

        assert get_vector_store(resources, "test_coll", embedding) is store
        assert get_vector_store(resources, "test_coll", embedding) is store

        mock_init_vdb.assert_called_once_with(
            resources=resources,
            collection_name="test_coll",
            embedding=embedding,
        )


@pytest.mark.anyio
async def test_delete_documents_by_source_uses_async_client():
    client = MagicMock()
    client.delete = AsyncMock()

    await delete_documents_by_source(client=client, collection_name="test_coll", source="test.pdf")

    client.delete.assert_awaited_once()
    call = client.delete.call_args.kwargs
    assert call["collection_name"] == "test_coll"
    condition = call["points_selector"].filter.must[0]
    assert condition.key == "metadata.source"
    assert condition.match.value == "test.pdf"


@pytest.mark.anyio
@patch("agent.utils.vdb.initialize_vector_db_async", new_callable=AsyncMock)
async def test_initialize_all_vector_dbs_uses_async_client(mock_init_vdb):
    config = Config()
    client = MagicMock()

    await initialize_all_vector_dbs(config, client)

    mock_init_vdb.assert_awaited_once_with(
        client=client,
        collection_name=config.qdrant_collection_name,
        embeddings_size=config.embedding_size,
    )
