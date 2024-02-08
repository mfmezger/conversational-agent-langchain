"""Strategy Pattern."""
from abc import ABC, abstractmethod

from agent.data_model.request_data_model import RAGRequest, SearchRequest


class LLMBase(ABC):
    """The LLM Base Strategy."""

    @abstractmethod
    def __init__(self, token: str, collection_name: str):
        """Init the LLM Base."""
        self.token = token
        self.collection_name = collection_name

    @abstractmethod
    def embed_documents(self, directory: str):
        """Embedd new docments in the Qdrant DB."""
        pass

    @abstractmethod
    def create_collection(self, name: str):
        """Create a new collection in the Vector Database."""
        pass

    @abstractmethod
    def search(self, search_request: SearchRequest):
        """Searches the documents in the Qdrant DB with semantic search."""
        pass

    @abstractmethod
    def rag(self, rag_request: RAGRequest):
        """Retrieval Augmented Generation."""
        pass

    @abstractmethod
    def summarize_text(self, text: str) -> str:
        """Summarize text."""
        pass
