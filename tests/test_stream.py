from __future__ import annotations

import os

import pytest
import requests


pytestmark = pytest.mark.e2e


@pytest.mark.skipif(os.getenv("RUN_LIVE_E2E") != "1", reason="Set RUN_LIVE_E2E=1 to run live stream test against localhost:8001")
def test_stream_live_backend() -> None:
    response = requests.post(
        "http://localhost:8001/rag/stream",
        json={"messages": [{"role": "user", "content": "Hello, how are you?"}], "collection_name": "default"},
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=30,
    )
    response.raise_for_status()
    assert response.status_code == 200
