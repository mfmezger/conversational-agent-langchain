"""Tests for the utility functions."""
from agent.utils.utility import generate_prompt


def test_generate_prompt() -> None:
    """Test that generate_prompt returns the correct prompt."""
    text = "blubby"
    query = "blubby2"

    prompt = generate_prompt(prompt_name="aleph_alpha_qa.j2", text=text, query=query, language="de")
    # assert prompt contains text and query
    assert text in prompt
    assert query in prompt


def test_generate_prompt_detect_language() -> None:
    """Test that generate_prompt returns the correct prompt."""
    text = "Das ist ein Stein der da am Wegrand steht."

    prompt = generate_prompt(prompt_name="aleph_alpha_qa.j2", text=text, language="detect")

    assert text in prompt


def test_generate_prompt_detect_language_default_parameter() -> None:
    """Test that generate_prompt returns the correct prompt."""
    text = "What is the capital of capital?"

    prompt = generate_prompt(prompt_name="aleph_alpha_qa.j2", text=text)

    assert text in prompt


def test_combine_text_from_list() -> None:
    """Test that combine_text_from_list returns the correct text."""
