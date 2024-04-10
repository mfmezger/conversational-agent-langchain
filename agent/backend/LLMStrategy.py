"""The Strategy Pattern for the LLM Backend."""
from agent.backend.aleph_alpha_service import AlephAlphaService
from agent.backend.gpt4all_service import GPT4AllService
from agent.backend.LLMBase import LLMBase
from agent.backend.open_ai_service import OpenAIService
from agent.data_model.request_data_model import LLMProvider, RAGRequest, SearchRequest


class LLMStrategyFactory:

    """The Factory to select the correct LLM Backend.

    Raises
    ------
        ValueError: If Provider is not known.

    Returns
    -------
        Strategy: The correct strategy.
    """

    _strategies: dict[str, type[LLMBase]] = {
        LLMProvider.ALEPH_ALPHA: AlephAlphaService,
        LLMProvider.OPENAI: OpenAIService,
        LLMProvider.GPT4ALL: GPT4AllService,
    }

    @staticmethod
    def get_strategy(strategy_type: str, token: str, collection_name: str) -> LLMBase:
        """Get the correct strategy.

        Args:
        ----
            strategy_type (str): The strategy type.
            token (str): The token for the strategy.
            collection_name (str): The collection name of the vector database.

        Raises:
        ------
            ValueError: _description_

        Returns:
        -------
            LLMBase: _description_
        """
        strategy = LLMStrategyFactory._strategies.get(strategy_type)
        if strategy is None:
            msg = "Unknown Strategy Type"
            raise ValueError(msg)
        return strategy(token, collection_name)


class LLMContext:

    """The Context for the LLM Backend."""

    def __init__(self, llm: LLMBase) -> None:
        """Init the Context."""
        self.llm = llm

    def change_strategy(self, strategy_type: str, token: str, collection_name: str) -> None:
        """Changes the strategy using the Factory."""
        self.llm = LLMStrategyFactory.get_strategy(strategy_type, token, collection_name)

    def search(self, search_request: SearchRequest) -> list:
        """Wrapper for the search."""
        return self.llm.search(search_request)

    def embed_documents(self, directory: str) -> None:
        """Wrapper for the Embedding of Documents."""
        return self.llm.embed_documents(directory)

    def create_collection(self, name: str) -> None:
        """Wrapper for creating a collection."""
        return self.llm.create_collection(name)

    def generate(self, prompt: str) -> str:
        """Wrapper for the generation of text."""
        return self.llm.generate(prompt)

    def rag(self, rag_request: RAGRequest) -> tuple:
        """Wrapper for the RAG."""
        return self.llm.rag(rag_request)

    def summarize_text(self, text: str) -> str:
        """Wrapper for the summarization of text."""
        return self.llm.summarize_text(text)


# Usage
# rag = LLMContext(LLMStrategyFactory.get_strategy(LLMProvider.ALEPH_ALPHA))

# rag.change_strategy(LLMProvider.GPT4ALL)