from __future__ import annotations

import importlib
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAI

pytestmark = pytest.mark.contract


class FakeConfiguredGraph:
    def __init__(self, parent: FakeGraph) -> None:
        self.parent = parent

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.parent.payloads.append(payload)
        if self.parent.fail_invoke:
            raise RuntimeError("private backend detail")
        return {"messages": [SimpleNamespace(content="RAG answer")], "documents": []}

    async def astream_events(self, payload: dict[str, Any], *, version: str) -> AsyncIterator[dict[str, Any]]:
        self.parent.payloads.append(payload)
        assert version == "v2"
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "grader"},
            "data": {"chunk": SimpleNamespace(content="ignored")},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "response_synthesizer"},
            "data": {"chunk": SimpleNamespace(content="RAG ")},
        }
        if self.parent.fail_stream:
            raise RuntimeError("stream failed")
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "response_synthesizer"},
            "data": {"chunk": SimpleNamespace(content="answer")},
        }


class FakeGraph:
    def __init__(self) -> None:
        self.configs: list[dict[str, Any]] = []
        self.payloads: list[dict[str, Any]] = []
        self.fail_invoke = False
        self.fail_stream = False

    def with_config(self, config: dict[str, Any]) -> FakeConfiguredGraph:
        self.configs.append(config)
        return FakeConfiguredGraph(self)


@pytest.fixture
def fake_graph(app, monkeypatch: pytest.MonkeyPatch) -> FakeGraph:
    module = importlib.import_module("agent.routes.openai_compat")
    graph = FakeGraph()
    monkeypatch.setattr(module, "graph", graph)
    monkeypatch.setattr(module.settings, "openai_compatible_model", "rag-test")
    monkeypatch.setattr(module.settings, "qdrant_collection_name", "configured-collection")
    return graph


@pytest.fixture
def sdk_client(client, fake_graph: FakeGraph) -> OpenAI:
    return OpenAI(api_key="test-key", base_url="http://testserver/v1", http_client=client)


def test_openapi_documents_compatibility_subset(app) -> None:
    schema = app.openapi()

    for path in ("/v1/chat/completions", "/v1/responses"):
        operation = schema["paths"][path]["post"]
        assert "422" not in operation["responses"]
        assert "text/event-stream" in operation["responses"]["200"]["content"]
        assert operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]


def test_chat_completion_contract(client, fake_graph: FakeGraph) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "rag-test",
            "messages": [
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
                {"role": "user", "content": "Current question"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "rag-test"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "RAG answer"},
            "finish_reason": "stop",
        }
    ]
    assert "usage" not in body
    assert fake_graph.configs == [{"metadata": {"collection_name": "configured-collection"}}]
    assert fake_graph.payloads[0]["messages"][-1] == {"role": "user", "content": "Current question"}


def test_chat_completion_stream_contract(client, fake_graph: FakeGraph) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "rag-test", "messages": [{"role": "user", "content": "Question"}], "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    data_lines = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert data_lines[-1] == "[DONE]"
    assert '"finish_reason":"stop"' in data_lines[-2]
    assert '"content":"RAG "' in data_lines[1]
    assert '"content":"answer"' in data_lines[2]
    assert fake_graph.configs == [{"metadata": {"collection_name": "configured-collection"}}]


def test_responses_contract_with_string_input(client, fake_graph: FakeGraph) -> None:
    response = client.post("/v1/responses", json={"model": "rag-test", "input": "Question"})

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["model"] == "rag-test"
    assert body["output"] == [
        {
            "id": body["output"][0]["id"],
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "RAG answer", "annotations": []}],
        }
    ]
    assert body["parallel_tool_calls"] is False
    assert body["tool_choice"] == "none"
    assert body["tools"] == []
    assert "usage" not in body
    assert fake_graph.configs == [{"metadata": {"collection_name": "configured-collection"}}]


def test_responses_stream_contract(client, fake_graph: FakeGraph) -> None:
    response = client.post(
        "/v1/responses",
        json={
            "model": "rag-test",
            "input": [
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
                {"role": "user", "content": "Current question"},
            ],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    event_names = [line.removeprefix("event: ") for line in response.text.splitlines() if line.startswith("event: ")]
    assert event_names == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert response.text.endswith("\n\n")
    assert fake_graph.configs == [{"metadata": {"collection_name": "configured-collection"}}]


@pytest.mark.parametrize(
    ("path", "body", "param"),
    [
        (
            "/v1/chat/completions",
            {"model": "rag-test", "messages": [{"role": "user", "content": "Question"}], "temperature": 0.2},
            "temperature",
        ),
        ("/v1/responses", {"model": "rag-test", "input": "Question", "tools": []}, "tools"),
        ("/v1/responses", {"model": "rag-test", "input": "Question", "instructions": "Be terse"}, "instructions"),
    ],
)
def test_unsupported_fields_use_openai_error_envelope(client, fake_graph: FakeGraph, path: str, body: dict[str, Any], param: str) -> None:
    response = client.post(path, json=body)

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": f"Unsupported field: `{param}`.",
            "type": "invalid_request_error",
            "param": param,
            "code": "unsupported_parameter",
        }
    }
    assert fake_graph.configs == []


def test_multimodal_input_is_rejected(client, fake_graph: FakeGraph) -> None:
    response = client.post(
        "/v1/responses",
        json={
            "model": "rag-test",
            "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.test/image"}]}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["code"] == "invalid_value"
    assert fake_graph.configs == []


def test_model_alias_is_validated_without_selecting_collection(client, fake_graph: FakeGraph) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "another-collection", "messages": [{"role": "user", "content": "Question"}]},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "The model `another-collection` does not exist or is not available.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }
    assert fake_graph.configs == []


def test_final_message_must_be_user(client, fake_graph: FakeGraph) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "rag-test", "messages": [{"role": "assistant", "content": "No question"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_value"
    assert fake_graph.configs == []


def test_openai_internal_error_does_not_leak_details(client, fake_graph: FakeGraph) -> None:
    fake_graph.fail_invoke = True
    response = client.post(
        "/v1/responses",
        json={"model": "rag-test", "input": "Question"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "message": "Internal server error.",
            "type": "server_error",
            "param": None,
            "code": "internal_error",
        }
    }


def test_unknown_v1_route_uses_openai_error_envelope(client, fake_graph: FakeGraph) -> None:
    response = client.get("/v1/not-implemented")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "Not Found",
            "type": "invalid_request_error",
            "param": None,
            "code": "not_found",
        }
    }


def test_non_openai_validation_error_is_unchanged(client, fake_graph: FakeGraph) -> None:
    response = client.post("/semantic/search", json={})

    assert response.status_code == 422
    assert "detail" in response.json()


def test_installed_sdk_parses_chat_non_streaming_and_streaming(sdk_client: OpenAI) -> None:
    completion = sdk_client.chat.completions.create(
        model="rag-test",
        messages=[{"role": "user", "content": "Question"}],
    )
    assert completion.object == "chat.completion"
    assert completion.choices[0].message.content == "RAG answer"
    assert completion.usage is None

    chunks = list(
        sdk_client.chat.completions.create(
            model="rag-test",
            messages=[{"role": "user", "content": "Question"}],
            stream=True,
        )
    )
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "RAG answer"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_installed_sdk_parses_responses_non_streaming_and_streaming(sdk_client: OpenAI) -> None:
    response = sdk_client.responses.create(model="rag-test", input="Question")
    assert response.object == "response"
    assert response.output_text == "RAG answer"
    assert response.usage is None

    raw_events = list(sdk_client.responses.create(model="rag-test", input="Question", stream=True))
    assert "".join(event.delta for event in raw_events if event.type == "response.output_text.delta") == "RAG answer"
    assert raw_events[-1].type == "response.completed"
    assert raw_events[-1].response.output_text == "RAG answer"

    with sdk_client.responses.stream(model="rag-test", input="Question") as stream:
        deltas = [event.delta for event in stream if event.type == "response.output_text.delta"]
        final_response = stream.get_final_response()
    assert "".join(deltas) == "RAG answer"
    assert final_response.output_text == "RAG answer"


@pytest.mark.anyio
async def test_midstream_failure_has_no_chat_done_marker(app, fake_graph: FakeGraph) -> None:
    module = importlib.import_module("agent.routes.openai_compat")
    schema_module = importlib.import_module("agent.data_model.openai_compat")
    fake_graph.fail_stream = True
    response = await module.create_chat_completion(
        schema_module.ChatCompletionRequest(
            model="rag-test",
            messages=[{"role": "user", "content": "Question"}],
            stream=True,
        )
    )

    chunks: list[str] = []
    with pytest.raises(RuntimeError, match="stream failed"):
        async for chunk in response.body_iterator:
            chunks.append(chunk)
    assert "[DONE]" not in "".join(chunks)
    assert '"finish_reason":"stop"' not in "".join(chunks)


@pytest.mark.anyio
async def test_midstream_failure_has_no_responses_completion_event(app, fake_graph: FakeGraph) -> None:
    module = importlib.import_module("agent.routes.openai_compat")
    schema_module = importlib.import_module("agent.data_model.openai_compat")
    fake_graph.fail_stream = True
    response = await module.create_response(
        schema_module.ResponsesRequest(model="rag-test", input="Question", stream=True)
    )

    chunks: list[str] = []
    with pytest.raises(RuntimeError, match="stream failed"):
        async for chunk in response.body_iterator:
            chunks.append(chunk)
    assert "response.completed" not in "".join(chunks)
