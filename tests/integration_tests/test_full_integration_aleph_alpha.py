"""Tests for the Aleph Alpha services."""
import os

import pytest
from dotenv import load_dotenv

from agent.backend.aleph_alpha_service import (
    embedd_text_aleph_alpha,
    embedd_text_files_aleph_alpha,
    get_db_connection,
    search_documents_aleph_alpha,
    send_completion_request,
    summarize_text_aleph_alpha,
)

load_dotenv()


def test_send_completion_request() -> None:
    """Test that send_completion_request returns a non-empty string."""
    prompt = "What is the meaning of life? A:"
    token = os.getenv("ALEPH_ALPHA_API_KEY")

    # make sure the token is not empty
    assert token is not None

    # Test that send_completion_request returns a non-empty string
    completion = send_completion_request(prompt, token)
    assert isinstance(completion, str)
    assert len(completion) > 0


def test_embedd_text_aleph_alpha() -> None:
    """Test that embedd_text_aleph_alpha does not raise an error."""
    # assert that it does not raise an error
    token = os.getenv("ALEPH_ALPHA_API_KEY")
    assert token is not None
    embedd_text_aleph_alpha("This is a test", "file", token, "+++")


def test_search_documents_aleph_alpha_wrong_token() -> None:
    """Test that search_documents_aleph_alpha raises an error when the token is invalid."""
    # Test with an empty query
    aleph_alpha_token = "example_token"
    query = ""
    with pytest.raises(ValueError, match="Query cannot be None or empty."):
        search_documents_aleph_alpha(aleph_alpha_token, query)

    # Test with an empty token
    aleph_alpha_token = ""
    query = "This should raise an error."
    with pytest.raises(ValueError, match="Token cannot be None or empty."):
        search_documents_aleph_alpha(aleph_alpha_token, query)


def test_db_connection() -> None:
    """Test that get_db_connection returns a non-empty connection."""
    assert get_db_connection(aleph_alpha_token="") is not None


def test_embedd_text_files_aleph_alpha() -> None:
    """Tests that embedd_text_files_aleph_alpha does not raise an error."""
    token = os.getenv("ALEPH_ALPHA_API_KEY")
    assert token is not None

    embedd_text_files_aleph_alpha(folder="tests/resources", aleph_alpha_token=token, seperator="###")


# def test_explain_completion() -> None:
#     """Test that explain_completion does not raise an error."""
#     explain_completion("This is a test", " ", 0.7, str(os.getenv("ALEPH_ALPHA_API_KEY")))


def test_summarize_text_aleph_alpha() -> None:
    """Test that summarize_text_aleph_alpha does not raise an error."""
    token = os.getenv("ALEPH_ALPHA_API_KEY")
    assert token is not None

    summarize_text_aleph_alpha(text="This is a test", token=str(os.getenv("ALEPH_ALPHA_API_KEY")))
