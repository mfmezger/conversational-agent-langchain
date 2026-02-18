from __future__ import annotations

import os

import pytest
from openai import OpenAI
from openai import OpenAIError


pytestmark = [pytest.mark.vcr, pytest.mark.integration]


OPENAI_KEY = os.getenv("OPENAI_API_KEY", "dummy_key")
LIVE_RECORD_MODES = {"all", "new_episodes", "rewrite"}


def _requires_live_openai_key() -> bool:
    record_mode = os.getenv("VCR_RECORD_MODE", "once")
    return record_mode in LIVE_RECORD_MODES and OPENAI_KEY == "dummy_key"


def test_openai_chat_completion_vcr() -> None:
    if _requires_live_openai_key():
        pytest.skip("Set OPENAI_API_KEY to record live OpenAI cassettes")

    client = OpenAI(api_key=OPENAI_KEY)
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


def test_openai_embeddings_vcr() -> None:
    if _requires_live_openai_key():
        pytest.skip("Set OPENAI_API_KEY to record live OpenAI cassettes")

    client = OpenAI(api_key=OPENAI_KEY)
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    try:
        response = client.embeddings.create(model=model, input="embedding smoke test")
    except OpenAIError as exc:  # pragma: no cover - depends on external account state
        pytest.skip(f"OpenAI embedding test skipped due to provider error: {exc}")

    assert response.data
    assert len(response.data[0].embedding) > 0
