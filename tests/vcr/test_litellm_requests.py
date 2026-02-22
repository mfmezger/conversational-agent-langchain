from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_litellm import ChatLiteLLM
from litellm.exceptions import NotFoundError


pytestmark = [pytest.mark.vcr, pytest.mark.integration]


GEMINI_KEY = os.getenv("GEMINI_API_KEY", "dummy_key")
LIVE_RECORD_MODES = {"all", "new_episodes", "rewrite"}
CASSETTE_PATH = Path("tests/vcr/cassettes/test_litellm_requests/test_litellm_gemini_chat_vcr.yaml")


def _requires_live_gemini_key() -> bool:
    record_mode = os.getenv("VCR_RECORD_MODE", "once")
    return record_mode in LIVE_RECORD_MODES and GEMINI_KEY == "dummy_key"


def test_litellm_gemini_chat_vcr() -> None:
    if _requires_live_gemini_key():
        pytest.skip("Set GEMINI_API_KEY to record live Gemini cassettes")

    if not CASSETTE_PATH.exists() and GEMINI_KEY == "dummy_key":
        pytest.skip("Gemini cassette is not recorded yet. Record once with GEMINI_API_KEY set.")

    model = os.getenv("GEMINI_TEST_MODEL", "gemini/gemini-3-flash-preview")
    llm = ChatLiteLLM(model_name=model, temperature=0)
    try:
        response = llm.invoke("Reply with exactly one short greeting.")
    except NotFoundError as exc:
        pytest.skip(f"Gemini model not available for this account/API: {exc}")

    assert isinstance(response.content, str)
    assert response.content.strip()
