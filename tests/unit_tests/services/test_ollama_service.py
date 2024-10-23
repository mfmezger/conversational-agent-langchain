import pytest
from unittest.mock import patch, MagicMock
from src.agent.backend.services.ollama_service import OllamaService

class TestOllamaService:
    @pytest.fixture
    def ollama_service(self):
        return OllamaService()

    @patch('src.agent.backend.services.ollama_service.requests')
    def test_generate_text(self, mock_requests, ollama_service):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Generated text"}
        mock_requests.post.return_value = mock_response

        result = ollama_service.generate_text("Test prompt")
        assert result == "Generated text"
        mock_requests.post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": "Test prompt", "stream": False}
        )

    @patch('src.agent.backend.services.ollama_service.requests')
    def test_generate_embedding(self, mock_requests, ollama_service):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_requests.post.return_value = mock_response

        result = ollama_service.generate_embedding("Test text")
        assert result == [0.1, 0.2, 0.3]
        mock_requests.post.assert_called_once_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "llama2", "prompt": "Test text"}
        )

    # Add more tests for other methods in OllamaService if they exist
