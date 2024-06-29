"""Strategy Pattern."""
from abc import ABC, abstractmethod

from agent.data_model.request_data_model import LLMBackend, SearchParams


class LLMBase(ABC):

    """The LLM Base Strategy."""

    @abstractmethod
    def __init__(self, collection_name: str | None) -> None:
        """Init the LLM Base."""
        self.collection_name = collection_name

    @abstractmethod
    def embed_documents(self, directory: str, llm_backend: LLMBackend) -> None:
        """Embedd new docments in the Qdrant DB."""

    @abstractmethod
    def create_collection(self, name: str) -> bool:
        """Create a new collection in the Vector Database."""

    @abstractmethod
    def create_search_chain(self, search: SearchParams) -> list:
        """Searches the documents in the Qdrant DB with semantic search."""

    @abstractmethod
    def summarize_text(self, text: str) -> str:
        """Summarize text."""
