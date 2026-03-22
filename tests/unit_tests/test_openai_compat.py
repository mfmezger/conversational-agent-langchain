"""Tests for the OpenAI-compatible /chat/completions endpoint."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def _mock_graph():
    """Patch the graph used by rag routes with a controllable mock."""
    fake_ai_message = MagicMock()
    fake_ai_message.content = "Paris is the capital of France."

    fake_doc = MagicMock()
    fake_doc.page_content = "France info"
    fake_doc.metadata = {"source": "wiki.pdf"}

    fake_result = {"messages": [fake_ai_message], "documents": [fake_doc]}

    mock_compiled = MagicMock()
    mock_configured = MagicMock()
    mock_configured.ainvoke = AsyncMock(return_value=fake_result)
    mock_compiled.with_config.return_value = mock_configured

    with patch("agent.routes.rag.graph", mock_compiled):
        yield mock_compiled, mock_configured


class TestOpenAIChatCompletionsNonStreaming:
    def test_basic_request(self, client, _mock_graph) -> None:
        response = client.post(
            "/rag/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "default"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Paris is the capital of France."
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["id"].startswith("chatcmpl-")
        # usage field should not be present (not tracked)
        assert "usage" not in data

    def test_collection_name_passed_to_graph(self, client, _mock_graph) -> None:
        mock_compiled, _ = _mock_graph
        client.post(
            "/rag/chat/completions",
            json={
                "model": "my-collection",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        mock_compiled.with_config.assert_called_once()
        call_kwargs = mock_compiled.with_config.call_args
        assert call_kwargs[0][0]["metadata"]["collection_name"] == "my-collection"

    def test_invalid_model_name_rejected(self, client) -> None:
        response = client.post(
            "/rag/chat/completions",
            json={
                "model": "../etc/passwd",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 422

    def test_empty_model_defaults_to_default(self, client, _mock_graph) -> None:
        """When model is omitted, it defaults to 'default'."""
        response = client.post(
            "/rag/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200
        assert response.json()["model"] == "default"

    def test_error_returns_openai_error_format(self, client) -> None:
        mock_configured = MagicMock()
        mock_configured.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        mock_compiled = MagicMock()
        mock_compiled.with_config.return_value = mock_configured

        with patch("agent.routes.rag.graph", mock_compiled):
            response = client.post(
                "/rag/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
        assert data["error"]["type"] == "server_error"


class TestOpenAIChatCompletionsStreaming:
    def test_stream_format(self, client, _mock_graph) -> None:
        mock_compiled, _ = _mock_graph

        # Set up streaming mock
        fake_chunk = {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "response_synthesizer"},
            "data": {"chunk": MagicMock(content="Hello world")},
        }

        async def fake_astream_events(*args, **kwargs):
            yield fake_chunk

        mock_stream_configured = MagicMock()
        mock_stream_configured.astream_events = fake_astream_events
        mock_compiled.with_config.return_value = mock_stream_configured

        response = client.post(
            "/rag/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = [line for line in response.text.strip().split("\n\n") if line.startswith("data:")]
        assert len(lines) >= 3  # initial + content + final + DONE

        # First chunk should have role
        first = json.loads(lines[0].removeprefix("data: "))
        assert first["choices"][0]["delta"]["role"] == "assistant"
        assert first["object"] == "chat.completion.chunk"

        # Content chunk
        content_chunk = json.loads(lines[1].removeprefix("data: "))
        assert content_chunk["choices"][0]["delta"]["content"] == "Hello world"

        # Final chunk
        final = json.loads(lines[2].removeprefix("data: "))
        assert final["choices"][0]["finish_reason"] == "stop"

        # DONE marker
        assert lines[-1] == "data: [DONE]"

    def test_stream_error_yields_error_event(self, client) -> None:
        async def failing_astream_events(*args, **kwargs):
            raise RuntimeError("stream boom")
            yield  # noqa: unreachable — makes this an async generator

        mock_configured = MagicMock()
        mock_configured.astream_events = failing_astream_events
        mock_compiled = MagicMock()
        mock_compiled.with_config.return_value = mock_configured

        with patch("agent.routes.rag.graph", mock_compiled):
            response = client.post(
                "/rag/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
        assert response.status_code == 200
        lines = [line for line in response.text.strip().split("\n\n") if line.startswith("data:")]
        # Should contain an error data event
        error_lines = [line for line in lines if "error" in line and "stream boom" not in line]
        assert any("server_error" in line for line in lines)


class TestModelValidation:
    @pytest.mark.parametrize(
        "model_name",
        ["default", "my-collection", "test_collection_123", "A1"],
    )
    def test_valid_model_names(self, client, _mock_graph, model_name) -> None:
        response = client.post(
            "/rag/chat/completions",
            json={"model": model_name, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "model_name",
        ["../traversal", "foo bar", "hello;drop", "", "-starts-with-dash"],
    )
    def test_invalid_model_names(self, client, model_name) -> None:
        response = client.post(
            "/rag/chat/completions",
            json={"model": model_name, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 422
