"""Unit tests for the conversational memory service."""

import pytest
from unittest.mock import MagicMock, patch


class TestConversationalMemory:
    """Tests for the ConversationalMemory class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock()
        config.qdrant_url = "http://localhost"
        config.qdrant_port = 6333
        config.qdrant_api_key = "test_key"
        config.model_name = "gpt-4"
        config.memory_enabled = True
        return config

    @pytest.fixture
    def memory_service(self, mock_config):
        """Create a memory service with mocked mem0."""
        with patch("agent.backend.services.memory_service.Memory") as mock_memory_cls:
            mock_memory = MagicMock()
            mock_memory_cls.from_config.return_value = mock_memory

            from agent.backend.services.memory_service import ConversationalMemory
            service = ConversationalMemory(config=mock_config)

            # Attach the mock for assertions
            service._mock_memory = mock_memory
            yield service

    def test_init_creates_memory_with_correct_config(self, mock_config):
        """Test that initialization creates mem0 with correct Qdrant config."""
        with patch("agent.backend.services.memory_service.Memory") as mock_memory_cls:
            mock_memory_cls.from_config.return_value = MagicMock()

            from agent.backend.services.memory_service import ConversationalMemory
            ConversationalMemory(config=mock_config)

            # Verify from_config was called with correct structure
            call_args = mock_memory_cls.from_config.call_args[0][0]
            assert call_args["vector_store"]["provider"] == "qdrant"
            assert call_args["vector_store"]["config"]["url"] == "http://localhost:6333"
            assert call_args["vector_store"]["config"]["api_key"] == "test_key"
            assert call_args["llm"]["provider"] == "litellm"
            assert call_args["embedder"]["provider"] == "openai"

    def test_search_returns_memories(self, memory_service):
        """Test that search returns a list of memory strings."""
        memory_service._mock_memory.search.return_value = {
            "results": [
                {"memory": "User prefers Python"},
                {"memory": "User is building a chatbot"},
            ]
        }

        result = memory_service.search(
            query="What programming language?",
            user_id="user_123"
        )

        assert result == ["User prefers Python", "User is building a chatbot"]
        memory_service._mock_memory.search.assert_called_once_with(
            query="What programming language?",
            user_id="user_123",
            run_id=None,
            agent_id=None,
            limit=5,
        )

    def test_search_with_all_scopes(self, memory_service):
        """Test search with user, session, and agent IDs."""
        memory_service._mock_memory.search.return_value = {"results": []}

        memory_service.search(
            query="test",
            user_id="user_1",
            session_id="session_1",
            agent_id="agent_1",
            limit=10,
        )

        memory_service._mock_memory.search.assert_called_once_with(
            query="test",
            user_id="user_1",
            run_id="session_1",  # mem0 uses run_id for session
            agent_id="agent_1",
            limit=10,
        )

    def test_search_handles_error(self, memory_service):
        """Test that search returns empty list on error."""
        memory_service._mock_memory.search.side_effect = Exception("Connection failed")

        result = memory_service.search(query="test", user_id="user_123")

        assert result == []

    def test_add_stores_messages(self, memory_service):
        """Test that add stores messages and returns result."""
        memory_service._mock_memory.add.return_value = {
            "results": [{"id": "mem_1", "memory": "stored memory"}]
        }

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = memory_service.add(
            messages=messages,
            user_id="user_123",
            session_id="session_abc",
        )

        assert len(result["results"]) == 1
        memory_service._mock_memory.add.assert_called_once_with(
            messages,
            user_id="user_123",
            run_id="session_abc",
            agent_id=None,
            metadata={},
        )

    def test_add_with_metadata(self, memory_service):
        """Test that add passes metadata correctly."""
        memory_service._mock_memory.add.return_value = {"results": []}

        memory_service.add(
            messages=[{"role": "user", "content": "test"}],
            user_id="user_1",
            metadata={"source": "api", "topic": "coding"},
        )

        call_args = memory_service._mock_memory.add.call_args
        assert call_args.kwargs["metadata"] == {"source": "api", "topic": "coding"}

    def test_add_handles_error(self, memory_service):
        """Test that add returns empty results on error."""
        memory_service._mock_memory.add.side_effect = Exception("Storage failed")

        result = memory_service.add(
            messages=[{"role": "user", "content": "test"}],
            user_id="user_123",
        )

        assert result == {"results": []}

    def test_get_all_returns_all_memories(self, memory_service):
        """Test get_all returns all memories for a scope."""
        memory_service._mock_memory.get_all.return_value = {
            "results": [
                {"memory": "Memory 1"},
                {"memory": "Memory 2"},
                {"memory": "Memory 3"},
            ]
        }

        result = memory_service.get_all(user_id="user_123")

        assert result == ["Memory 1", "Memory 2", "Memory 3"]
        memory_service._mock_memory.get_all.assert_called_once_with(
            user_id="user_123",
            run_id=None,
            agent_id=None,
        )

    def test_get_all_handles_error(self, memory_service):
        """Test that get_all returns empty list on error."""
        memory_service._mock_memory.get_all.side_effect = Exception("Failed")

        result = memory_service.get_all(user_id="user_123")

        assert result == []

    def test_delete_by_memory_id(self, memory_service):
        """Test delete by specific memory ID."""
        result = memory_service.delete(memory_id="mem_123")

        assert result is True
        memory_service._mock_memory.delete.assert_called_once_with(memory_id="mem_123")

    def test_delete_all_by_user(self, memory_service):
        """Test delete all memories for a user."""
        result = memory_service.delete(user_id="user_123")

        assert result is True
        memory_service._mock_memory.delete_all.assert_called_once_with(
            user_id="user_123",
            run_id=None,
            agent_id=None,
        )

    def test_delete_handles_error(self, memory_service):
        """Test that delete returns False on error."""
        memory_service._mock_memory.delete.side_effect = Exception("Delete failed")

        result = memory_service.delete(memory_id="mem_123")

        assert result is False


class TestGetMemoryService:
    """Tests for the get_memory_service singleton function."""

    def test_get_memory_service_returns_singleton(self):
        """Test that get_memory_service returns the same instance."""
        with patch("agent.backend.services.memory_service.Memory") as mock_memory_cls:
            mock_memory_cls.from_config.return_value = MagicMock()

            # Reset the singleton
            import agent.backend.services.memory_service as memory_module
            memory_module._memory_instance = None

            from agent.backend.services.memory_service import get_memory_service

            service1 = get_memory_service()
            service2 = get_memory_service()

            assert service1 is service2

            # Clean up
            memory_module._memory_instance = None
