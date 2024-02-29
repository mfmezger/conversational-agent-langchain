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
