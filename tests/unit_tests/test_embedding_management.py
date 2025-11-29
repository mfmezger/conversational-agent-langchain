import pytest
from unittest.mock import MagicMock, patch
from agent.backend.services.embedding_management import EmbeddingManagement
from agent.data_model.request_data_model import SearchParams
from langchain_core.documents import Document

@pytest.fixture
def mock_config():
    with patch("agent.backend.services.embedding_management.config") as mock_cfg:
        mock_cfg.embedding_provider = "cohere"
        mock_cfg.embedding_model_name = "embed-english-v3.0"
        mock_cfg.embedding_size = 1024
        yield mock_cfg

@pytest.fixture
def mock_init_vdb():
    with patch("agent.backend.services.embedding_management.init_vdb") as mock_vdb:
        yield mock_vdb

@pytest.fixture
def mock_cohere_embeddings():
    with patch("agent.backend.services.embedding_management.CohereEmbeddings") as mock_embed:
        yield mock_embed

@pytest.fixture
def embedding_service(mock_config, mock_init_vdb, mock_cohere_embeddings):
    return EmbeddingManagement(collection_name="test_collection")

def test_init_success(mock_config, mock_init_vdb, mock_cohere_embeddings):
    service = EmbeddingManagement(collection_name="test_collection")

    assert service.collection_name == "test_collection"
    mock_cohere_embeddings.assert_called_once_with(model="embed-english-v3.0")
    mock_init_vdb.assert_called_once()

def test_init_invalid_provider(mock_config):
    mock_config.embedding_provider = "unknown_provider"

    with pytest.raises(KeyError, match="No suitable embedding Model configured!"):
        EmbeddingManagement(collection_name="test_collection")

@patch("agent.backend.services.embedding_management.DirectoryLoader")
@patch("agent.backend.services.embedding_management.RecursiveCharacterTextSplitter")
def test_embed_documents_pdf(mock_splitter_cls, mock_loader_cls, embedding_service):
    # Setup mocks
    mock_loader = MagicMock()
    mock_loader_cls.return_value = mock_loader

    mock_doc = Document(page_content="test content", metadata={"source": "/path/to/test.pdf"})
    mock_loader.load_and_split.return_value = [mock_doc]

    # Call method
    embedding_service.embed_documents(directory="tests/resources/", file_ending=".pdf")

    # Assertions
    mock_loader_cls.assert_called_once()
    mock_loader.load_and_split.assert_called_once()
    embedding_service.vector_db.add_texts.assert_called_once()

    # Verify metadata processing (source path cleanup)
    call_args = embedding_service.vector_db.add_texts.call_args
    assert call_args is not None
    metadatas = call_args.kwargs.get("metadatas") or call_args[1] # handle both keyword and positional args if needed, though code uses kwargs
    assert metadatas[0]["source"] == "test.pdf"

@patch("agent.backend.services.embedding_management.DirectoryLoader")
@patch("agent.backend.services.embedding_management.RecursiveCharacterTextSplitter")
def test_embed_documents_txt(mock_splitter_cls, mock_loader_cls, embedding_service):
    # Setup mocks
    mock_loader = MagicMock()
    mock_loader_cls.return_value = mock_loader
    mock_loader.load_and_split.return_value = []

    # Call method
    embedding_service.embed_documents(directory="tests/resources/", file_ending=".txt")

    # Assertions
    mock_loader_cls.assert_called_once()

def test_embed_documents_invalid_extension(embedding_service):
    with pytest.raises(ValueError, match="File ending not supported."):
        embedding_service.embed_documents(directory="tests/resources/", file_ending=".docx")

@patch("agent.backend.services.embedding_management.generate_collection")
def test_create_collection(mock_generate_collection, embedding_service):
    result = embedding_service.create_collection(name="new_collection")

    assert result is True
    mock_generate_collection.assert_called_once_with("new_collection", 1024)
