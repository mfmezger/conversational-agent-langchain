import pytest
from unittest.mock import patch, MagicMock
from src.agent.backend.services.open_ai_service import OpenAIService

class TestOpenAIService:
    @pytest.fixture
    def openai_service(self):
        return OpenAIService()

    @patch('src.agent.backend.services.open_ai_service.openai')
    def test_generate_text(self, mock_openai, openai_service):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(text="Generated text")]
        mock_openai.Completion.create.return_value = mock_response

        result = openai_service.generate_text("Test prompt")
        assert result == "Generated text"
        mock_openai.Completion.create.assert_called_once_with(
            engine="text-davinci-002",
            prompt="Test prompt",
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )

    @patch('src.agent.backend.services.open_ai_service.openai')
    def test_generate_embedding(self, mock_openai, openai_service):
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_openai.Embedding.create.return_value = mock_response

        result = openai_service.generate_embedding("Test text")
        assert result == [0.1, 0.2, 0.3]
        mock_openai.Embedding.create.assert_called_once_with(
            input="Test text",
            model="text-embedding-ada-002"
        )

    # Add more tests for other methods in OpenAIService if they exist
