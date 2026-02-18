from __future__ import annotations

import os

import pytest
from langchain_core.documents import Document

from agent.utils.reranker import rerank_with_cohere


pytestmark = [pytest.mark.vcr, pytest.mark.integration]


COHERE_KEY = os.getenv("COHERE_API_KEY", "dummy_key")
LIVE_RECORD_MODES = {"all", "new_episodes", "rewrite"}


def _requires_live_cohere_key() -> bool:
    record_mode = os.getenv("VCR_RECORD_MODE", "once")
    return record_mode in LIVE_RECORD_MODES and COHERE_KEY == "dummy_key"


def test_langchain_cohere_embedding_vcr() -> None:
    if _requires_live_cohere_key():
        pytest.skip("Set COHERE_API_KEY to record live Cohere cassettes")

    from langchain_cohere import CohereEmbeddings

    model = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
    embeddings = CohereEmbeddings(model=model, cohere_api_key=COHERE_KEY)

    vector = embeddings.embed_query("embedding smoke test")

    assert vector
    assert len(vector) > 0


def test_cohere_reranker_vcr() -> None:
    if _requires_live_cohere_key():
        pytest.skip("Set COHERE_API_KEY to record live Cohere cassettes")

    docs = [
        Document(page_content="Paris is the capital of France."),
        Document(page_content="The moon orbits Earth."),
    ]

    model = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    result = rerank_with_cohere(docs, "What is the capital of France?", top_k=1, api_key=COHERE_KEY, model=model)

    assert len(result) == 1
    assert "France" in result[0].page_content
