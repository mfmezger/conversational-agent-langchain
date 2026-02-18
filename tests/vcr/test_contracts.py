from __future__ import annotations

from unittest.mock import patch

import pytest
from inline_snapshot import snapshot

from tests.fakes.rag import FakeAsyncRetriever, FakeDoc


pytestmark = [pytest.mark.vcr, pytest.mark.integration]


def test_search_contract_with_vcr_marker(client) -> None:
    retriever = FakeAsyncRetriever([FakeDoc(page_content="hello", metadata={"page": 1, "source": "doc.txt"})])
    with patch("agent.routes.search.get_retriever", return_value=retriever):
        response = client.post("/semantic/search", json={"query": "hello", "collection_name": "default", "k": 2})

    assert response.status_code == 200
    assert response.json() == snapshot([{"text": "hello", "page": 1, "source": "doc.txt"}])
