from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from agent.backend.services.embedding_management import EmbeddingManagement
from agent.utils.vdb import VDBResources


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.embedding_provider = "cohere"
    config.embedding_model_name = "embed-english-v3.0"
    config.embedding_size = 1024
    return config


@pytest.fixture
def resources(mock_config):
    return VDBResources(
        config=mock_config,
        sync_client=MagicMock(),
        async_client=MagicMock(),
        sparse_embeddings=MagicMock(),
    )


@pytest.fixture
def mock_init_vdb():
    with patch("agent.backend.services.embedding_management.init_vdb") as mock_vdb:
        yield mock_vdb


@pytest.fixture
def mock_get_embedding_model():
    with patch("agent.backend.services.embedding_management.get_embedding_model") as mock_embed:
        mock_embed.return_value = MagicMock()
        yield mock_embed


@pytest.fixture
def embedding_service(resources, mock_init_vdb, mock_get_embedding_model):
    return EmbeddingManagement(collection_name="test_collection", resources=resources)


def test_init_success(mock_config, resources, mock_init_vdb, mock_get_embedding_model):
    service = EmbeddingManagement(collection_name="test_collection", resources=resources)

    assert service.collection_name == "test_collection"
    mock_get_embedding_model.assert_called_once_with(mock_config)
    mock_init_vdb.assert_called_once_with(
        resources=resources,
        collection_name="test_collection",
        embedding=mock_get_embedding_model.return_value,
    )


def test_init_invalid_provider(mock_config, resources):
    mock_config.embedding_provider = "unknown_provider"

    with pytest.raises(KeyError, match="No suitable embedding Model configured!"):
        EmbeddingManagement(collection_name="test_collection", resources=resources)


@patch("agent.backend.services.embedding_management.DirectoryLoader")
@patch("agent.backend.services.embedding_management.RecursiveCharacterTextSplitter")
def test_embed_documents_pdf(mock_splitter_cls, mock_loader_cls, embedding_service):
    mock_loader = MagicMock()
    mock_loader_cls.return_value = mock_loader

    mock_doc = Document(page_content="test content", metadata={"source": "/path/to/test.pdf"})
    mock_loader.load_and_split.return_value = [mock_doc]

    embedding_service.embed_documents(directory="tests/resources/", file_ending=".pdf")

    mock_loader_cls.assert_called_once()
    mock_loader.load_and_split.assert_called_once()
    embedding_service.vector_db.add_texts.assert_called_once()

    call_args = embedding_service.vector_db.add_texts.call_args
    assert call_args is not None
    metadatas = call_args.kwargs["metadatas"]
    assert metadatas[0]["source"] == "test.pdf"


@patch("agent.backend.services.embedding_management.DirectoryLoader")
@patch("agent.backend.services.embedding_management.RecursiveCharacterTextSplitter")
def test_embed_documents_txt(mock_splitter_cls, mock_loader_cls, embedding_service):
    mock_loader = MagicMock()
    mock_loader_cls.return_value = mock_loader
    mock_loader.load_and_split.return_value = []

    embedding_service.embed_documents(directory="tests/resources/", file_ending=".txt")

    mock_loader_cls.assert_called_once()


def test_embed_documents_invalid_extension(embedding_service):
    with pytest.raises(ValueError, match="File ending not supported."):
        embedding_service.embed_documents(directory="tests/resources/", file_ending=".docx")


@patch("agent.backend.services.embedding_management.generate_collection")
def test_create_collection(mock_generate_collection, embedding_service, resources):
    result = embedding_service.create_collection(name="new_collection")

    assert result is True
    mock_generate_collection.assert_called_once_with(
        client=resources.sync_client,
        collection_name="new_collection",
        embeddings_size=1024,
    )
