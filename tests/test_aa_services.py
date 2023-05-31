"""Tests for the Aleph Alpha services."""
import os

import pytest
from dotenv import load_dotenv

from agent.backend.aleph_alpha_service import generate_prompt, send_completion_request

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
