import pytest
from unittest.mock import patch, MagicMock
from src.agent.backend.services.cohere_service import CohereService

class TestCohereService:
    @pytest.fixture
    def cohere_service(self):
        return CohereService()

    @patch('src.agent.backend.services.cohere_service.cohere')
    def test_generate_text(self, mock_cohere, cohere_service):
        mock_response = MagicMock()
        mock_response.generations = [MagicMock(text="Generated text")]
        mock_cohere.generate.return_value = mock_response

        result = cohere_service.generate_text("Test prompt")
        assert result == "Generated text"
        mock_cohere.generate.assert_called_once_with(model='command', prompt="Test prompt", max_tokens=256, temperature=0.75)

    @patch('src.agent.backend.services.cohere_service.cohere')
    def test_generate_embedding(self, mock_cohere, cohere_service):
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_cohere.embed.return_value = mock_response

        result = cohere_service.generate_embedding("Test text")
        assert result == [0.1, 0.2, 0.3]
        mock_cohere.embed.assert_called_once_with(texts=["Test text"], model="embed-english-v2.0")

    # Add more tests for other methods in CohereService if they exist
