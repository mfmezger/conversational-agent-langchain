"""Strategy Pattern."""
from abc import ABC, abstractmethod

from agent.data_model.request_data_model import Filtering, LLMBackend, RAGRequest, SearchParams


class LLMBase(ABC):

    """The LLM Base Strategy."""

    @abstractmethod
    def __init__(self, token: str | None, collection_name: str | None) -> None:
        """Init the LLM Base."""
        self.token = token
        self.collection_name = collection_name

    @abstractmethod
    def embed_documents(self, directory: str, llm_backend: LLMBackend) -> None:
        """Embedd new docments in the Qdrant DB."""

    @abstractmethod
    def create_collection(self, name: str) -> bool:
        """Create a new collection in the Vector Database."""

    @abstractmethod
    def search(self, search: SearchParams, filtering: Filtering) -> list:
        """Searches the documents in the Qdrant DB with semantic search."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""

    @abstractmethod
    def rag(self, rag: RAGRequest, search: SearchParams, filtering: Filtering) -> tuple:
        """Retrieval Augmented Generation."""

    @abstractmethod
    def summarize_text(self, text: str) -> str:
        """Summarize text."""
