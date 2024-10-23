"""The Strategy Pattern for the LLM Backend.

This module implements the Strategy and Factory patterns for managing different
LLM (Language Model) backends. It provides a flexible way to switch between
different LLM providers and perform operations like searching, embedding
documents, and summarizing text.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Type
from functools import lru_cache
from loguru import logger

from agent.backend.LLMBase import LLMBase
from agent.backend.services.cohere_service import CohereService
from agent.backend.services.ollama_service import OllamaService
from agent.backend.services.open_ai_service import OpenAIService
from agent.data_model.request_data_model import LLMProvider, SearchParams

# Set up logging

class LLMStrategyFactory:
    """A factory class for creating LLM backend strategies.

    This class uses a class variable to store the mapping between LLMProvider
    enum values and their corresponding service classes.
    """

    _strategies: ClassVar[Dict[LLMProvider, Type[LLMBase]]] = {
        LLMProvider.OPENAI: OpenAIService,
        LLMProvider.COHERE: CohereService,
        LLMProvider.OLLAMA: OllamaService,
    }

    @staticmethod
    @lru_cache(maxsize=32)
    def get_strategy(provider: LLMProvider, collection_name: str) -> LLMBase:
        """Get the correct LLM strategy based on the provider.

        This method is cached to improve performance for repeated calls with the same arguments.

        Args:
            provider (LLMProvider): The LLM provider enum.
            collection_name (str): The collection name of the vector database.

        Raises:
            ValueError: If the provider is not supported.

        Returns:
            LLMBase: An instance of the appropriate LLM service.
        """
        strategy_class = LLMStrategyFactory._strategies.get(provider)
        if strategy_class is None:
            logger.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        logger.info(f"Creating strategy for provider: {provider}")
        return strategy_class(collection_name=collection_name)

    @classmethod
    def register_strategy(cls, provider: LLMProvider, strategy_class: Type[LLMBase]) -> None:
        """Register a new LLM strategy.

        Args:
            provider (LLMProvider): The LLM provider enum.
            strategy_class (Type[LLMBase]): The class implementing the LLM strategy.
        """
        cls._strategies[provider] = strategy_class
        logger.info(f"Registered new strategy for provider: {provider}")


class LLMOperation(ABC):
    """Abstract base class for LLM operations.

    This class defines the interface for all LLM operations.
    """

    @abstractmethod
    def execute(self, llm: LLMBase, *args: Any, **kwargs: Any) -> Any:
        """Execute the LLM operation.

        Args:
            llm (LLMBase): The LLM backend to use for the operation.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the operation, type varies depending on the specific operation.
        """


class SearchOperation(LLMOperation):
    """LLM operation for searching."""

    def execute(self, llm: LLMBase, search: SearchParams) -> list:
        """Execute a search operation.

        Args:
            llm (LLMBase): The LLM backend to use for the operation.
            search (SearchParams): The search parameters.

        Returns:
            list: The search results.
        """
        return llm.create_search_chain(search=search)


class EmbedDocumentsOperation(LLMOperation):
    """LLM operation for embedding documents."""

    def execute(self, llm: LLMBase, directory: str, file_ending: str) -> None:
        """Execute a document embedding operation.

        Args:
            llm (LLMBase): The LLM backend to use for the operation.
            directory (str): The directory containing the documents to embed.
            file_ending (str): The file extension of the documents to embed.

        Returns:
            None
        """
        return llm.embed_documents(directory=directory, file_ending=file_ending)


class CreateCollectionOperation(LLMOperation):
    """LLM operation for creating a collection."""

    def execute(self, llm: LLMBase, name: str) -> None:
        """Execute a collection creation operation.

        Args:
            llm (LLMBase): The LLM backend to use for the operation.
            name (str): The name of the collection to create.

        Returns:
            None
        """
        return llm.create_collection(name)


class SummarizeTextOperation(LLMOperation):
    """LLM operation for summarizing text."""

    def execute(self, llm: LLMBase, text: str) -> str:
        """Execute a text summarization operation.

        Args:
            llm (LLMBase): The LLM backend to use for the operation.
            text (str): The text to summarize.

        Returns:
            str: The summarized text.
        """
        return llm.summarize_text(text)


class LLMContext:
    """The Context for the LLM Backend.

    This class manages the current LLM strategy and provides methods to perform
    various LLM operations.
    """

    def __init__(self, provider: LLMProvider, collection_name: str) -> None:
        """Initialize the LLMContext.

        Args:
            provider (LLMProvider): The initial LLM provider to use.
            collection_name (str): The collection name of the vector database.
        """
        self.llm = LLMStrategyFactory.get_strategy(provider, collection_name)
        logger.info(f"Initialized LLMContext with provider: {provider}")

    def change_strategy(self, provider: LLMProvider, collection_name: str) -> None:
        """Change the current LLM strategy.

        Args:
            provider (LLMProvider): The new LLM provider to use.
            collection_name (str): The collection name of the vector database.
        """
        self.llm = LLMStrategyFactory.get_strategy(provider, collection_name)
        logger.info(f"Changed strategy to provider: {provider}")

    def execute_operation(self, operation: LLMOperation, *args: Any, **kwargs: Any) -> Any:
        """Execute an LLM operation using the current strategy.

        Args:
            operation (LLMOperation): The operation to execute.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the operation, type varies depending on the specific operation.
        """
        logger.info(f"Executing operation: {operation.__class__.__name__}")
        try:
            return operation.execute(self.llm, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing operation {operation.__class__.__name__}: {str(e)}")
            raise

    # Convenience methods for common operations
    def search(self, search: SearchParams) -> list:
        """Perform a search operation."""
        return self.execute_operation(SearchOperation(), search)

    def embed_documents(self, directory: str, file_ending: str) -> None:
        """Embed documents."""
        return self.execute_operation(EmbedDocumentsOperation(), directory, file_ending)

    def create_collection(self, name: str) -> None:
        """Create a collection."""
        return self.execute_operation(CreateCollectionOperation(), name)

    def summarize_text(self, text: str) -> str:
        """Summarize text."""
        return self.execute_operation(SummarizeTextOperation(), text)
