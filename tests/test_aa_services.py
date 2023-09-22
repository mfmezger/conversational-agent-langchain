"""Tests for the Aleph Alpha services."""
import os

import pytest
from dotenv import load_dotenv

from agent.backend.aleph_alpha_service import (
    explain_completion,
    generate_prompt,
    search_documents_aleph_alpha,
    send_completion_request,
)

load_dotenv()


def test_prompt_folder_not_empty() -> None:
    """Test that the prompts folder is not empty."""
    prompt_files = os.listdir("prompts")
    assert len(prompt_files) > 0


def test_generate_prompt() -> None:
    """Test that generate_prompt returns a non-empty string with the expected text and query."""
    text = "ASDF"
    query = "FDSA"
    prompt = generate_prompt(prompt_name="qa.j2", text=text, query=query)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert text in prompt
    assert query in prompt

    # Test that generate_prompt raises an error when the template file is not found
    with pytest.raises(FileNotFoundError):
        generate_prompt("nonexistent_template.j2", text=text, query=query)

    # Test that generate_prompt raises an error when the template file is invalid
    with pytest.raises(Exception):
        generate_prompt("invalid_template.j2", text=text, query=query)


def test_send_completion_request_empty() -> None:
    """Test with an empty prompt."""
    prompt = ""
    token = "example_token"
    with pytest.raises(ValueError, match="Text cannot be None or empty."):
        send_completion_request(prompt, token)

    # Test with an empty token
    prompt = "This should raise an error."
    token = ""
    with pytest.raises(ValueError, match="Token cannot be None or empty."):
        send_completion_request(prompt, token)


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


def test_search_documents_aleph_alpha_empty() -> None:
    """Test with an empty query or token."""
    aleph_alpha_token = "example_token"
    query = ""
    with pytest.raises(ValueError, match="Query cannot be None or empty."):
        search_documents_aleph_alpha(aleph_alpha_token, query)

    aleph_alpha_token = ""
    query = "This should raise an error."
    with pytest.raises(ValueError, match="Token cannot be None or empty."):
        search_documents_aleph_alpha(aleph_alpha_token, query)


def test_explain_completion() -> None:  # todo: rework
    """Test that explain_completion does not raise an error."""
    explain_completion("This is a test", " ", str(os.getenv("ALEPH_ALPHA_API_KEY")))


# def test_search_documents_aleph_alpha_empty() -> None:
# """Test that search_documents_aleph_alpha returns a list of tuples."""
# aleph_alpha_token = "example_token"
# query = "example_query"
# amount = 1

# # Test with an invalid amount
# with pytest.raises(PermissionError):
#     search_documents_aleph_alpha(aleph_alpha_token, query, amount)

# amount = 0
# # Test with an invalid amount
# with pytest.raises(ValueError):
#     search_documents_aleph_alpha(aleph_alpha_token, query, amount)
