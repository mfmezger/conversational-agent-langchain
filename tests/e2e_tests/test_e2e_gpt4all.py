"""Test the GPT4All service."""
import pytest
from langchain_community.vectorstores import Qdrant
from langchain_core.document import Document

from agent.backend.gpt4all_service import GPT4ALLService


@pytest.fixture()
def gpt4all() -> GPT4ALLService:
    """Return a GPT4ALL instance."""
    # first create a qdrant collection for testing
    return GPT4ALLService("gpt4all_test", "test_token")


def test_init(gpt4all: GPT4ALLService) -> None:
    """Test the init method."""
    assert isinstance(gpt4all, GPT4ALLService)


def test_get_db_connection(gpt4all: GPT4ALLService) -> None:
    """Test the get_db_connection method."""
    db = gpt4all.get_db_connection("test_collection")
    assert isinstance(db, Qdrant)


def test_embedd_documents_gpt4all(gpt4all: GPT4ALLService) -> None:
    """Test the embedd_documents_gpt4all method."""
    assert gpt4all.embedd_documents_gpt4all("test_dir") is None


def test_embedd_text_gpt4all(gpt4all: GPT4ALLService) -> None:
    """Test the embedd_text_gpt4all method."""
    assert gpt4all.embedd_text_gpt4all("test_text", "test_file", "test_seperator") is None


def test_summarize_text_gpt4all(gpt4all: GPT4ALLService) -> None:
    """Test the summarize_text_gpt4all method."""
    summary = gpt4all.summarize_text_gpt4all("test_text")
    assert isinstance(summary, str)


def test_completion_text_gpt4all(gpt4all: GPT4ALLService) -> None:
    """Test the completion_text_gpt4all method."""
    completion = gpt4all.completion_text_gpt4all("test_prompt")
    assert isinstance(completion, str)


def test_custom_completion_prompt_gpt4all(gpt4all: GPT4ALLService) -> None:
    """Test the custom_completion_prompt_gpt4all method."""
    completion = gpt4all.custom_completion_prompt_gpt4all("test_prompt")
    assert isinstance(completion, str)


def test_search_documents_gpt4all(gpt4all: GPT4ALLService) -> None:
    """Test the search_documents_gpt4all method."""
    results = gpt4all.search_documents_gpt4all("test_query", 1)
    assert isinstance(results, list)
    for result in results:
        assert isinstance(result, tuple)
        assert isinstance(result[0], Document)
        assert isinstance(result[1], float)


def test_rag_gpt4all(gpt4all: GPT4ALLService) -> None:
    """Test the qa_gpt4all method."""
    documents = [(Document("test_content", {"source": "test_source", "page": 0}), 0.5)]
    answer, prompt, meta_data = gpt4all.rag(documents, "test_query")
    assert isinstance(answer, str)
    assert isinstance(prompt, str)
    assert isinstance(meta_data, list)
