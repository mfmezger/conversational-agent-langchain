"""Test module for prompts.py functionality."""

import os
import pytest
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock
from omegaconf import DictConfig

from src.agent.utils.prompts import load_prompt_from_file, load_prompts


@pytest.fixture
def sample_prompt_content():
    return "This is a test prompt template\nWith multiple lines\n{variable}"


@pytest.fixture
def mock_config():
    return DictConfig({
        "prompts": {
            "response_template": "prompts/response_template.txt",
            "cohere_response_template": "prompts/cohere_response_template.txt",
            "rephrase_template": "prompts/rephrase_template.txt"
        }
    })


def test_load_prompt_from_file_success(tmp_path, sample_prompt_content):
    """Test successful loading of a prompt file."""
    # Create a temporary file with test content
    test_file = tmp_path / "test_prompt.txt"
    test_file.write_text(sample_prompt_content)

    # Load and verify content
    result = load_prompt_from_file(str(test_file))
    assert result == sample_prompt_content


def test_load_prompt_from_file_not_found():
    """Test handling of non-existent prompt file."""
    with pytest.raises(FileNotFoundError):
        load_prompt_from_file("nonexistent_file.txt")


def test_load_prompt_from_file_permission_error(tmp_path):
    """Test handling of permission error when reading prompt file."""
    test_file = tmp_path / "no_permission.txt"
    test_file.write_text("test content")

    # Remove read permissions
    os.chmod(test_file, 0o000)

    with pytest.raises(IOError):
        load_prompt_from_file(str(test_file))

    # Restore permissions for cleanup
    os.chmod(test_file, 0o666)


@patch('src.agent.utils.prompts.load_prompt_from_file')
def test_load_prompts_success(mock_load_prompt, mock_config):
    """Test successful loading of all prompt templates."""
    # Setup mock returns
    mock_load_prompt.side_effect = [
        "response template content",
        "cohere response template content",
        "rephrase template content"
    ]

    # Call function and verify results
    response, cohere, rephrase = load_prompts(mock_config)

    assert response == "response template content"
    assert cohere == "cohere response template content"
    assert rephrase == "rephrase template content"

    # Verify correct files were loaded
    assert mock_load_prompt.call_count == 3
    mock_load_prompt.assert_any_call("prompts/response_template.txt")
    mock_load_prompt.assert_any_call("prompts/cohere_response_template.txt")
    mock_load_prompt.assert_any_call("prompts/rephrase_template.txt")


@patch('src.agent.utils.prompts.load_prompt_from_file')
def test_load_prompts_missing_file(mock_load_prompt, mock_config):
    """Test handling of missing prompt file during loading."""
    mock_load_prompt.side_effect = FileNotFoundError("File not found")

    with pytest.raises(FileNotFoundError):
        load_prompts(mock_config)


@patch('src.agent.utils.prompts.load_prompt_from_file')
def test_load_prompts_io_error(mock_load_prompt, mock_config):
    """Test handling of IO error during prompt loading."""
    mock_load_prompt.side_effect = IOError("Permission denied")

    with pytest.raises(IOError):
        load_prompts(mock_config)


def test_load_prompts_invalid_config():
    """Test handling of invalid configuration."""
    invalid_config = DictConfig({"prompts": {}})

    with pytest.raises(AttributeError):
        load_prompts(invalid_config)
