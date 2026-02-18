"""Reranker utilities for document reranking."""

from typing import TYPE_CHECKING, Literal

from langchain_core.documents import Document
from loguru import logger

if TYPE_CHECKING:
    from flashrank import Ranker

RerankerProvider = Literal["cohere", "flashrank", "none"]

# Cache for FlashRank model (expensive to load)
_flashrank_ranker: "Ranker | None" = None


def _get_flashrank_ranker() -> "Ranker":
    """Get or create cached FlashRank ranker."""
    global _flashrank_ranker
    if _flashrank_ranker is None:
        from flashrank import Ranker  # noqa: PLC0415

        _flashrank_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
        logger.info("FlashRank model loaded and cached")
    return _flashrank_ranker


def rerank_with_cohere(
    documents: list[Document],
    query: str,
    top_k: int,
    api_key: str,
    model: str = "rerank-v3.5",
) -> list[Document]:
    """Rerank documents using Cohere Rerank API.

    Args:
        documents: List of documents to rerank.
        query: The query to rerank against.
        top_k: Number of top documents to return.
        api_key: Cohere API key.
        model: Cohere rerank model name.

    Returns:
        Reranked list of documents.

    """
    from langchain_cohere import CohereRerank  # noqa: PLC0415

    if not documents:
        return documents

    reranker = CohereRerank(model=model, cohere_api_key=api_key, top_n=top_k)
    reranked = reranker.compress_documents(documents=documents, query=query)
    logger.info(f"Cohere reranked {len(documents)} documents to top {len(reranked)}")
    return list(reranked)


def rerank_with_flashrank(documents: list[Document], query: str, top_k: int) -> list[Document]:
    """Rerank documents using FlashRank (local model).

    Args:
        documents: List of documents to rerank.
        query: The query to rerank against.
        top_k: Number of top documents to return.

    Returns:
        Reranked list of documents.

    """
    from flashrank import RerankRequest  # noqa: PLC0415

    if not documents:
        return documents

    ranker = _get_flashrank_ranker()

    # Convert documents to flashrank format
    passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(documents)]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    # Sort by score and take top_k
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    # Reconstruct documents preserving original metadata
    reranked_docs = []
    for result in sorted_results:
        original_idx = result["id"]
        reranked_docs.append(documents[original_idx])

    logger.info(f"FlashRank reranked {len(documents)} documents to top {len(reranked_docs)}")
    return reranked_docs


def get_reranker(
    provider: RerankerProvider,
    top_k: int = 3,
    cohere_api_key: str | None = None,
) -> callable:
    """Get a reranker function based on the provider.

    Args:
        provider: The reranker provider to use.
        top_k: Number of top documents to return after reranking.
        cohere_api_key: Cohere API key (required if provider is "cohere").

    Returns:
        A callable that takes (documents, query) and returns reranked documents.

    """
    match provider:
        case "none":
            logger.info("Reranking disabled, using passthrough")
            return lambda docs, _: docs[:top_k] if len(docs) > top_k else docs

        case "cohere":
            if not cohere_api_key:
                msg = "Cohere API key is required for Cohere reranker"
                raise ValueError(msg)
            return lambda docs, query: rerank_with_cohere(docs, query, top_k, cohere_api_key)

        case "flashrank":
            # Pre-warm the model on startup
            _get_flashrank_ranker()
            return lambda docs, query: rerank_with_flashrank(docs, query, top_k)

        case _:
            msg = f"Unknown reranker provider: {provider}"
            raise ValueError(msg)
