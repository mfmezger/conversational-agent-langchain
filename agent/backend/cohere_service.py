"""Cohere Backend."""
from dotenv import load_dotenv

from agent.backend.LLMBase import LLMBackend, LLMBase
from agent.data_model.request_data_model import (
    Filtering,
    RAGRequest,
    SearchRequest,
)

load_dotenv()


class CohereService(LLMBase):

    """Wrapper for cohere llms."""

    def embed_documents(self, directory: str, llm_backend: LLMBackend) -> None:
        """Embedd new docments in the Qdrant DB."""

    def create_collection(self, name: str) -> bool:
        """Create a new collection in the Vector Database."""

    def search(self, search: SearchRequest, filtering: Filtering) -> list:
        """Searches the documents in the Qdrant DB with semantic search."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""

    def rag(self, rag: RAGRequest, search: SearchRequest, filtering: Filtering) -> tuple:
        """Retrieval Augmented Generation."""

    def summarize_text(self, text: str) -> str:
        """Summarize text."""
