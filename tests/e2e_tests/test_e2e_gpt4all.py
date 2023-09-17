"""End 2 End Tests for the GPT4ALL Component."""
from agent.backend.gpt4all_service import (
    completion_text_gpt4all,
    embedd_documents_gpt4all,
    search_documents_gpt4all,
    summarize_text_gpt4all,
)


def test_embedd_text_gpt4all() -> None:
    """Test that embedd_text_gpt4all does not raise an error."""
    # assert that it does not raise an error
    embedd_documents_gpt4all(dir="tests/resources")


def test_summarize_text_gpt4all() -> None:
    """Testing the summarization."""
    summary = summarize_text_gpt4all(text="Das ist ein Test.")
    # assert not empty
    assert len(summary) > 0


def test_completion_text_gpt4all() -> None:
    """Testing the completion."""
    completion = completion_text_gpt4all(prompt="What is AI?")
    # assert not empty
    assert len(completion) > 0


def test_search_gpt4all() -> None:
    """Testing the search."""
    search_results = search_documents_gpt4all(query="Was ist Vanille?", amount=3)

    assert len(search_results) > 0
    assert isinstance(search_results, list)
