"""End 2 End Tests for the GPT4ALL Component."""
from agent.backend.gpt4all_service import (
    completion_text_gpt4all,
    embedd_documents_gpt4all,
    summarize_text_gpt4all,
)


def test_embedd_text_gpt4all():
    """Test that embedd_text_gpt4all does not raise an error."""
    # assert that it does not raise an error
    embedd_documents_gpt4all("This is a test")


def test_summarize_text_gpt4all():
    """Testing the summarization."""
    summary = summarize_text_gpt4all(text="Das ist ein Test.")
    # assert not empty
    assert len(summary) > 0


def test_completion_text_gpt4all():
    """Testing the completion."""
    completion = completion_text_gpt4all(text="Das ist ein Test.", query="Was ist das?")
    # assert not empty
    assert len(completion) > 0
