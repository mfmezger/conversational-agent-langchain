"""Strategy Pattern."""
from abc import ABC, abstractmethod

from agent.data_model.request_data_model import RAGRequest, SearchRequest


class LLMBase(ABC):

    """The LLM Base Strategy."""

    @abstractmethod
    def __init__(self, token: str, collection_name: str) -> None:
        """Init the LLM Base."""
        self.token = token
        self.collection_name = collection_name

    @abstractmethod
    def embed_documents(self, directory: str) -> None:
        """Embedd new docments in the Qdrant DB."""

    @abstractmethod
    def create_collection(self, name: str) -> None:
        """Create a new collection in the Vector Database."""

    @abstractmethod
    def search(self, search_request: SearchRequest) -> list:
        """Searches the documents in the Qdrant DB with semantic search."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""

    @abstractmethod
    def rag(self, rag_request: RAGRequest) -> tuple:
        """Retrieval Augmented Generation."""

    @abstractmethod
    def summarize_text(self, text: str) -> str:
        """Summarize text."""


class LLMStrategyFactory:
    """The Factory to select the correct LLM Backend.

    Raises:
        ValueError: If Provider is not known.

    Returns:
        Strategy: The correct strategy. 
    """
    _strategies: Dict[str, Type[LLMBase]] = {
        LLMProvider.ALEPH_ALPHA: AlephAlphaService,
        LLMProvider.OPENAI: OpenAIService,
        LLMProvider.GPT4ALL: GPT4ALLService,
    }

    @staticmethod
    def get_strategy(strategy_type: str, token: str, collection_name: str) -> LLMBase:
        """Get the correct strategy.

        Args:
            strategy_type (str): The strategy type.
            token (str): The token for the strategy.
            collection_name (str): The collection name of the vector database.

        Raises:
            ValueError: _description_

        Returns:
            LLMBase: _description_
        """
        Strategy = LLMStrategyFactory._strategies.get(strategy_type)
        if Strategy is None:
            raise ValueError("Unknown Strategy Type")
        return Strategy(token, collection_name)


class LLMContext:
    def __init__(self, llm: LLMBase) -> None:
        self.llm = llm

    def change_strategy(self, strategy_type: str, token: str, collection_name: str):
        self.llm = LLMStrategyFactory.get_strategy(strategy_type, token, collection_name)

    def search(self, search_request: SearchRequest) -> list:
        return self.llm.search(search_request)

    def embed_documents(self, directory: str):
        return self.llm.embed_documents(directory)

    def create_collection(self, name: str):
        return self.llm.create_collection(name)

    def generate(self, prompt: str) -> str:
        return self.llm.generate(prompt)

    def rag(self, rag_request: RAGRequest) -> tuple:
        return self.llm.rag(rag_request)

    def summarize_text(self, text: str) -> str:
        return self.llm.summarize_text(text)


# Usage
# rag = LLMContext(LLMStrategyFactory.get_strategy(LLMProvider.ALEPH_ALPHA))

# rag.change_strategy(LLMProvider.GPT4ALL)
