"""Tests for the Aleph Alpha services."""
import os

import pytest
from dotenv import load_dotenv

from agent.backend.aleph_alpha_service import (
    generate_prompt,
    get_db_connection,
    search_documents_aleph_alpha,
    send_completion_request,
)

load_dotenv()


def test_prompt_folder_not_empty():
    """Test that the prompts folder is not empty."""
    prompt_files = os.listdir("prompts")
    assert len(prompt_files) > 0


def test_generate_prompt():
    """Test that generate_prompt returns a non-empty string with the expected text and query."""
    text = "ASDF"
    query = "FDSA"
    prompt = generate_prompt("qa.j2", text=text, query=query)
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


def test_send_completion_request_empty():
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


def test_send_completion_request():
    """Test that send_completion_request returns a non-empty string."""
    prompt = "What is the meaning of life? A:"
    token = os.getenv("ALEPH_ALPHA_API_KEY")

    # make sure the token is not empty
    assert token is not None

    # Test that send_completion_request returns a non-empty string
    completion = send_completion_request(prompt, token)
    assert isinstance(completion, str)
    assert len(completion) > 0


def test_search_documents_aleph_alpha_wrong_token():
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


def test_db_connection():
    """Test that get_db_connection returns a non-empty connection."""
    assert get_db_connection(aleph_alpha_token="") is not None


def test_search_documents_aleph_alpha_empty():
    """Test with an empty query or token."""
    aleph_alpha_token = "example_token"
    query = ""
    with pytest.raises(ValueError, match="Query cannot be None or empty."):
        search_documents_aleph_alpha(aleph_alpha_token, query)

    aleph_alpha_token = ""
    query = "This should raise an error."
    with pytest.raises(ValueError, match="Token cannot be None or empty."):
        search_documents_aleph_alpha(aleph_alpha_token, query)


def test_search_documents_aleph_alpha():
    """Test that search_documents_aleph_alpha returns a list of tuples."""
    aleph_alpha_token = "example_token"
    query = "example_query"
    amount = 1

    # Test with an invalid amount
    with pytest.raises(PermissionError):
        search_documents_aleph_alpha(aleph_alpha_token, query, amount)

    amount = 0
    # Test with an invalid amount
    with pytest.raises(ValueError):
        search_documents_aleph_alpha(aleph_alpha_token, query, amount)
