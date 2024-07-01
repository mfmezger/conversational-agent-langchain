from unittest.mock import patch

import pytest
from cohere_service import CohereService
from omegaconf import DictConfig

from agent.data_model.request_data_model import SearchParams


@pytest.fixture()
def mock_cfg():
    return DictConfig({"qdrant": {"collection_name_cohere": "test_collection"}, "cohere_embeddings": {"embedding_model_name": "test-model", "size": 768}})


@pytest.fixture()
def cohere_service_instance(mock_cfg):
    with patch("cohere_service.load_dotenv"), patch("cohere_service.load_config", return_value=mock_cfg):
        return CohereService(cfg=mock_cfg, collection_name=None)


def test_cohere_service_initialization(cohere_service_instance) -> None:
    assert cohere_service_instance.collection_name == "test_collection"
    assert cohere_service_instance.cfg.cohere_embeddings.embedding_model_name == "test-model"


@patch("cohere_service.DirectoryLoader")
@patch("cohere_service.PyPDFium2Loader")
@patch("cohere_service.TextLoader")
def test_embed_documents(mock_text_loader, mock_pdf_loader, mock_dir_loader, cohere_service_instance) -> None:
    cohere_service_instance.embed_documents(directory="tests/resources/", file_ending=".pdf")
    mock_dir_loader.assert_called_once()
    # Further assertions can be added to verify the behavior


def test_create_collection(cohere_service_instance) -> None:
    with patch("cohere_service.generate_collection") as mock_generate_collection:
        result = cohere_service_instance.create_collection(name="new_collection")
        mock_generate_collection.assert_called_once_with("new_collection", 768)
        assert result is True


def test_create_search_chain(cohere_service_instance) -> None:
    search_params = SearchParams(query="test query", k=5)
    with patch("cohere_service.chain") as mock_chain:
        cohere_service_instance.create_search_chain(search=search_params)
        # Verify the chain decorator was used
        mock_chain.assert_called_once()
        # Further assertions can be added to verify the retriever's behavior


# Additional tests can be added for other methods and edge cases
