"""Test the GPT4All service."""
import pytest

from agent.backend.gpt4all_service import GPT4ALLService


@pytest.fixture()
def gpt4all() -> GPT4ALLService:
    """Return a GPT4ALL instance."""
    # first create a qdrant collection for testing
    return GPT4ALLService("gpt4all_test", "test_token")


def test_init(gpt4all: GPT4ALLService) -> None:
    """Test the init method."""
    assert isinstance(gpt4all, GPT4ALLService)
