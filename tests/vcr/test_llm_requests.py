from __future__ import annotations

import os

import pytest
from openai import OpenAI
from openai import OpenAIError


pytestmark = [pytest.mark.vcr, pytest.mark.integration]


LIVE_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
HAS_OPENAI_KEY = bool(LIVE_OPENAI_KEY) and LIVE_OPENAI_KEY != "dummy_key"


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="Requires OPENAI_API_KEY for recording/replay setup")
def test_openai_chat_completion_vcr() -> None:
    client = OpenAI(api_key=LIVE_OPENAI_KEY)
    model = os.getenv("OPENAI_TEST_MODEL", "gpt-4o-mini")

    try:
        response = client.responses.create(
            model=model,
            input="Say hello in one short sentence.",
            max_output_tokens=30,
        )
    except OpenAIError as exc:  # pragma: no cover - depends on external account state
        pytest.skip(f"OpenAI test skipped due to provider error: {exc}")

    assert response.output_text
    assert isinstance(response.output_text, str)


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="Requires OPENAI_API_KEY for recording/replay setup")
def test_openai_embeddings_vcr() -> None:
    client = OpenAI(api_key=LIVE_OPENAI_KEY)
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    try:
        response = client.embeddings.create(model=model, input="embedding smoke test")
    except OpenAIError as exc:  # pragma: no cover - depends on external account state
        pytest.skip(f"OpenAI embedding test skipped due to provider error: {exc}")

    assert response.data
    assert len(response.data[0].embedding) > 0
