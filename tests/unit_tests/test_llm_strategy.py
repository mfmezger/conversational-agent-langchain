import pytest
from src.agent.backend.LLMStrategy import LLMStrategy
from src.agent.backend.services.cohere_service import CohereService
from src.agent.backend.services.ollama_service import OllamaService
from src.agent.backend.services.open_ai_service import OpenAIService

class TestLLMStrategy:
    def test_llm_strategy_initialization(self):
        strategy = LLMStrategy()
        assert isinstance(strategy, LLMStrategy)

    def test_set_strategy(self):
        strategy = LLMStrategy()
        
        strategy.set_strategy("cohere")
        assert isinstance(strategy.strategy, CohereService)
        
        strategy.set_strategy("ollama")
        assert isinstance(strategy.strategy, OllamaService)
        
        strategy.set_strategy("openai")
        assert isinstance(strategy.strategy, OpenAIService)

    def test_invalid_strategy(self):
        strategy = LLMStrategy()
        with pytest.raises(ValueError):
            strategy.set_strategy("invalid_service")

    @pytest.mark.asyncio
    async def test_generate_text(self):
        strategy = LLMStrategy()
        strategy.set_strategy("cohere")  # Assuming Cohere is the default
        
        # Mock the generate_text method of CohereService
        strategy.strategy.generate_text = lambda prompt: "Mocked response"
        
        result = await strategy.generate_text("Test prompt")
        assert result == "Mocked response"

    @pytest.mark.asyncio
    async def test_generate_embedding(self):
        strategy = LLMStrategy()
        strategy.set_strategy("cohere")  # Assuming Cohere is the default
        
        # Mock the generate_embedding method of CohereService
        strategy.strategy.generate_embedding = lambda text: [0.1, 0.2, 0.3]
        
        result = await strategy.generate_embedding("Test text")
        assert result == [0.1, 0.2, 0.3]

    # Add more tests for other methods in LLMStrategy if they exist
