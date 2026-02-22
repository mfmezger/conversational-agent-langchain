from __future__ import annotations

import importlib
import os
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


ALLOWED_TEST_HOSTS: set[str] = {"localhost", "127.0.0.1", "::1", "testserver"}
VCR_REDACTED_HEADERS: set[str] = {
    "authorization",
    "api-key",
    "x-api-key",
    "x-goog-api-key",
    "cookie",
    "set-cookie",
    "openai-organization",
    "openai-project",
    "x-request-id",
    "cf-ray",
    "x-debug-trace-id",
}


def _is_allowed_host(url: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname
    # Relative URLs should be allowed.
    if host is None:
        return True
    return host in ALLOWED_TEST_HOSTS


@pytest.fixture(scope="session")
def anyio_backend() -> Literal["asyncio"]:
    return "asyncio"


@pytest.fixture(autouse=True, scope="session")
def test_env_defaults() -> None:
    os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://dummy.openai.azure.com/")
    os.environ.setdefault("AZURE_OPENAI_API_KEY", "dummy_key")
    os.environ.setdefault("OPENAI_API_KEY", "dummy_key")
    os.environ.setdefault("COHERE_API_KEY", "dummy_key")
    os.environ.setdefault("GEMINI_API_KEY", "dummy_key")
    os.environ.setdefault("QDRANT_URL", "http://localhost")
    os.environ.setdefault("QDRANT_PORT", "6333")
    os.environ.setdefault("QDRANT_API_KEY", "test_api_key")


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": sorted(VCR_REDACTED_HEADERS),
        "filter_query_parameters": ["key", "api_key"],
        "before_record_request": _sanitize_vcr_request,
        "before_record_response": _sanitize_vcr_response,
        "decode_compressed_response": True,
        "record_mode": os.getenv("VCR_RECORD_MODE", "once"),
    }


def _sanitize_vcr_request(request: Any) -> Any:
    request.headers = _strip_sensitive_headers(dict(request.headers))
    return request


def _sanitize_vcr_response(response: dict[str, Any]) -> dict[str, Any]:
    response["headers"] = _strip_sensitive_headers(dict(response.get("headers", {})))
    return response


def _strip_sensitive_headers(headers: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in headers.items() if key.lower() not in VCR_REDACTED_HEADERS}


@pytest.fixture(autouse=True)
def block_external_http(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Block outgoing HTTP by default, except localhost/testserver.

    Set `ALLOW_NETWORK_TESTS=1` to bypass this guard.
    """
    if os.getenv("ALLOW_NETWORK_TESTS") == "1" or request.node.get_closest_marker("vcr"):
        return

    import httpx
    import requests

    original_sync_request = httpx.Client.request
    original_async_request = httpx.AsyncClient.request
    original_requests_request = requests.sessions.Session.request

    def guarded_sync_request(self: httpx.Client, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        if not _is_allowed_host(str(url)):
            raise RuntimeError(f"External HTTP blocked in tests: {url}")
        return original_sync_request(self, method, url, *args, **kwargs)

    async def guarded_async_request(self: httpx.AsyncClient, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        if not _is_allowed_host(str(url)):
            raise RuntimeError(f"External HTTP blocked in tests: {url}")
        return await original_async_request(self, method, url, *args, **kwargs)

    def guarded_requests_request(
        self: requests.sessions.Session, method: str, url: str, *args: Any, **kwargs: Any
    ) -> Any:
        if not _is_allowed_host(str(url)):
            raise RuntimeError(f"External HTTP blocked in tests: {url}")
        return original_requests_request(self, method, url, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "request", guarded_sync_request)
    monkeypatch.setattr(httpx.AsyncClient, "request", guarded_async_request)
    monkeypatch.setattr(requests.sessions.Session, "request", guarded_requests_request)


@pytest.fixture(scope="session")
def app() -> Iterator[FastAPI]:
    """Import the FastAPI app with expensive startup side effects patched out."""
    with ExitStack() as stack:
        stack.enter_context(patch("agent.utils.vdb.initialize_all_vector_dbs", return_value=None))
        stack.enter_context(patch("phoenix.otel.register", return_value=None))
        stack.enter_context(
            patch("openinference.instrumentation.langchain.LangChainInstrumentor.instrument", return_value=None)
        )

        module = importlib.import_module("agent.api")
        yield module.app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def resources_path() -> Path:
    return Path("tests/resources")
