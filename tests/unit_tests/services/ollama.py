from unittest.mock import MagicMock, patch

import pytest
from ollama_service import OllamaService
from omegaconf import DictConfig


@pytest.fixture()
def mock_cfg():
    return DictConfig({"qdrant": {"collection_name_ollama": "test_collection", "embeddings_size": 768}, "ollama_embeddings": {"embedding_model_name": "test_model"}})


@pytest.fixture()
def ollama_service(mock_cfg):
    with patch("ollama_service.load_config", return_value=mock_cfg):
        return OllamaService(cfg=mock_cfg, collection_name="test_collection")


def test_initialization(ollama_service) -> None:
    assert ollama_service.collection_name == "test_collection"
    assert ollama_service.cfg == mock_cfg()


@patch("ollama_service.DirectoryLoader")
@patch("ollama_service.PyPDFium2Loader")
@patch("ollama_service.TextLoader")
def test_embed_documents(mock_text_loader, mock_pdf_loader, mock_dir_loader, ollama_service) -> None:
    ollama_service.embed_documents(directory="tests/resources/", file_ending=".pdf")
    mock_dir_loader.assert_called_once()
    # Further assertions can be added to validate the behavior


def test_create_collection(ollama_service) -> None:
    with patch.object(ollama_service.vector_db, "add_texts", return_value=True):
        result = ollama_service.create_collection(name="new_collection")
        assert result is True
        # You can add more assertions here to check if the collection was created with the correct parameters


@patch("ollama_service.chain")
def test_create_search_chain(mock_chain, ollama_service) -> None:
    search_params = MagicMock()
    ollama_service.create_search_chain(search=search_params)
    mock_chain.assert_called_once()
    # Further assertions can be added to validate the search chain behavior
