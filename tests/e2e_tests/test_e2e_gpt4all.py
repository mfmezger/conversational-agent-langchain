"""Test the GPT4All service."""
import pytest

from agent.backend.gpt4all_service import GPT4AllService


@pytest.fixture()
def gpt4all() -> GPT4AllService:
    """Return a GPT4ALL instance."""
    # first create a qdrant collection for testing
    return GPT4AllService(collection_name="gpt4all_test", token="test_token")


def test_init(gpt4all: GPT4AllService) -> None:
    """Test the init method."""
    assert isinstance(gpt4all, GPT4AllService)
