import pytest
from src.agent.backend.LLMBase import LLMBase

class TestLLMBase:
    def test_llm_base_initialization(self):
        llm = LLMBase()
        assert isinstance(llm, LLMBase)

    def test_abstract_methods(self):
        llm = LLMBase()
        with pytest.raises(NotImplementedError):
            llm.generate_text("test prompt")
        
        with pytest.raises(NotImplementedError):
            llm.generate_embedding("test text")

    # Add more tests for any concrete methods in LLMBase if they exist
