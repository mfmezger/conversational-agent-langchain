from __future__ import annotations

import os

import pytest
from langchain_core.documents import Document

from agent.utils.reranker import rerank_with_cohere


pytestmark = [pytest.mark.vcr, pytest.mark.integration]


LIVE_COHERE_KEY = os.getenv("COHERE_API_KEY", "")
HAS_COHERE_KEY = bool(LIVE_COHERE_KEY) and LIVE_COHERE_KEY != "dummy_key"


@pytest.mark.skipif(not HAS_COHERE_KEY, reason="Requires COHERE_API_KEY for recording/replay setup")
def test_langchain_cohere_embedding_vcr() -> None:
    from langchain_cohere import CohereEmbeddings

    model = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
    embeddings = CohereEmbeddings(model=model, cohere_api_key=LIVE_COHERE_KEY)

    vector = embeddings.embed_query("embedding smoke test")

    assert vector
    assert len(vector) > 0


@pytest.mark.skipif(not HAS_COHERE_KEY, reason="Requires COHERE_API_KEY for recording/replay setup")
def test_cohere_reranker_vcr() -> None:
    docs = [
        Document(page_content="Paris is the capital of France."),
        Document(page_content="The moon orbits Earth."),
    ]

    model = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    result = rerank_with_cohere(docs, "What is the capital of France?", top_k=1, api_key=LIVE_COHERE_KEY, model=model)

    assert len(result) == 1
    assert "France" in result[0].page_content
