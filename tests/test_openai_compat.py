from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from openai import APIError, OpenAI

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
        try:
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
        finally:
            self.parent.stream_closed = True


class FakeGraph:
    def __init__(self) -> None:
        self.configs: list[dict[str, Any]] = []
        self.payloads: list[dict[str, Any]] = []
        self.fail_invoke = False
        self.fail_stream = False
        self.stream_closed = False

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

    components = schema["components"]["schemas"]
    assert set(components["ChatCompletionChoice"]["required"]) == {"index", "message", "logprobs", "finish_reason"}
    assert set(components["ChatCompletionResponseMessage"]["required"]) == {"role", "content", "refusal"}
    assert {
        "object",
        "completed_at",
        "error",
        "incomplete_details",
        "instructions",
        "metadata",
        "parallel_tool_calls",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
        "usage",
    } <= set(components["ResponsesResponse"]["required"])


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
            "message": {"role": "assistant", "content": "RAG answer", "refusal": None},
            "logprobs": None,
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
    chunks = [json.loads(line) for line in data_lines[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": "", "refusal": None}
    assert all(chunk["choices"][0]["logprobs"] is None for chunk in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[1]["choices"][0]["delta"]["content"] == "RAG "
    assert chunks[2]["choices"][0]["delta"]["content"] == "answer"
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
    required_fields = {
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "none",
        "tools": [],
        "top_p": None,
        "usage": None,
    }
    assert body["completed_at"] is not None
    for field, value in required_fields.items():
        assert body[field] == value
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


@pytest.mark.parametrize("stream", ["true", 1, None])
@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", {"model": "rag-test", "messages": [{"role": "user", "content": "Question"}]}),
        ("/v1/responses", {"model": "rag-test", "input": "Question"}),
    ],
)
def test_stream_requires_a_boolean(client, fake_graph: FakeGraph, path: str, body: dict[str, Any], stream: Any) -> None:
    response = client.post(path, json={**body, "stream": stream})

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "Input should be a valid boolean",
        "type": "invalid_request_error",
        "param": "stream",
        "code": "invalid_value",
    }
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


@pytest.mark.parametrize("path", ["/v1", "/v1/not-implemented"])
def test_unknown_v1_route_uses_openai_error_envelope(client, fake_graph: FakeGraph, path: str) -> None:
    response = client.get(path)

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


def test_chat_stream_failure_uses_sdk_error_contract(client, sdk_client: OpenAI, fake_graph: FakeGraph) -> None:
    fake_graph.fail_stream = True
    request = {
        "model": "rag-test",
        "messages": [{"role": "user", "content": "Question"}],
        "stream": True,
    }

    response = client.post("/v1/chat/completions", json=request)
    data = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert json.loads(data[-1]) == {
        "error": {
            "message": "Internal server error.",
            "type": "server_error",
            "param": None,
            "code": "internal_error",
        }
    }
    assert "[DONE]" not in data
    assert '"finish_reason":"stop"' not in response.text

    with pytest.raises(APIError, match="Internal server error"):
        list(
            sdk_client.chat.completions.create(
                model="rag-test",
                messages=[{"role": "user", "content": "Question"}],
                stream=True,
            )
        )
    assert fake_graph.stream_closed is True


def test_responses_stream_failure_uses_official_sdk_events(client, sdk_client: OpenAI, fake_graph: FakeGraph) -> None:
    fake_graph.fail_stream = True
    response = client.post("/v1/responses", json={"model": "rag-test", "input": "Question", "stream": True})
    event_names = [line.removeprefix("event: ") for line in response.text.splitlines() if line.startswith("event: ")]
    payloads = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")]

    assert event_names[-2:] == ["error", "response.failed"]
    assert "response.completed" not in event_names
    assert payloads[-2] == {
        "type": "error",
        "code": "server_error",
        "message": "Internal server error.",
        "param": None,
        "sequence_number": payloads[-2]["sequence_number"],
    }
    assert payloads[-1]["type"] == "response.failed"
    assert payloads[-1]["response"]["status"] == "failed"
    assert payloads[-1]["response"]["error"] == {"code": "server_error", "message": "Internal server error."}

    events = list(sdk_client.responses.create(model="rag-test", input="Question", stream=True))
    assert [event.type for event in events[-2:]] == ["error", "response.failed"]
    assert events[-2].code == "server_error"
    assert events[-1].response.status == "failed"
    assert events[-1].response.error.code == "server_error"

    with sdk_client.responses.stream(model="rag-test", input="Question") as stream:
        stream_events = list(stream)
        with pytest.raises(RuntimeError, match="Didn't receive a `response.completed` event"):
            stream.get_final_response()
    assert stream_events[-1].type == "response.failed"
    assert fake_graph.stream_closed is True
