"""Conversational memory service using mem0 for persistent, personalized context."""

from loguru import logger
from mem0 import Memory

from agent.utils.config import Config


class ConversationalMemory:
    """Manages long-term conversational memory using mem0.

    Provides multi-level memory (user, session, agent) with semantic search
    for context enrichment across conversations.
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the memory service.

        Args:
            config: Optional Config instance. Uses default if not provided.

        """
        self.cfg = config or Config()

        # Configure mem0 with Qdrant backend (self-hosted, reusing existing Qdrant instance)
        # Build the Qdrant URL with port
        qdrant_url = f"{self.cfg.qdrant_url}:{self.cfg.qdrant_port}"

        mem0_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "conversational_memory",
                    "url": qdrant_url,  # Use url instead of host/port to respect http vs https
                    "api_key": self.cfg.qdrant_api_key,
                },
            },
            "llm": {
                "provider": "litellm",
                "config": {
                    "model": self.cfg.model_name,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                },
            },
        }

        self.memory = Memory.from_config(mem0_config)
        logger.info("Conversational memory service initialized with Qdrant backend")

    def search(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> list[str]:
        """Retrieve relevant memories for context enrichment.

        Args:
            query: The search query (typically the user's current question).
            user_id: Optional user identifier for user-scoped memories.
            session_id: Optional session identifier for session-scoped memories.
            agent_id: Optional agent identifier for agent-scoped memories.
            limit: Maximum number of memories to return.

        Returns:
            List of memory strings relevant to the query.

        """
        try:
            results = self.memory.search(
                query=query,
                user_id=user_id,
                run_id=session_id,  # mem0 uses run_id for session scope
                agent_id=agent_id,
                limit=limit,
            )
            memories = [mem["memory"] for mem in results.get("results", [])]
            logger.debug(f"Retrieved {len(memories)} memories for query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            memories = []
        return memories

    def add(
        self,
        messages: list[dict],
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Store new memories from a conversation turn.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            user_id: Optional user identifier for user-scoped memories.
            session_id: Optional session identifier for session-scoped memories.
            agent_id: Optional agent identifier for agent-scoped memories.
            metadata: Optional additional metadata to store with memories.

        Returns:
            Result dict from mem0 with stored memory details.

        """
        try:
            result = self.memory.add(
                messages,
                user_id=user_id,
                run_id=session_id,
                agent_id=agent_id,
                metadata=metadata or {},
            )
            num_added = len(result.get("results", []))
            logger.debug(f"Stored {num_added} memories for user={user_id}, session={session_id}")
        except Exception as e:
            logger.error(f"Error storing memories: {e}")
            result = {"results": []}
        return result

    def get_all(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[str]:
        """Get all memories for a given scope.

        Args:
            user_id: Optional user identifier.
            session_id: Optional session identifier.
            agent_id: Optional agent identifier.

        Returns:
            List of all memory strings for the specified scope.

        """
        try:
            results = self.memory.get_all(
                user_id=user_id,
                run_id=session_id,
                agent_id=agent_id,
            )
            memories = [mem["memory"] for mem in results.get("results", [])]
        except Exception as e:
            logger.error(f"Error getting all memories: {e}")
            memories = []
        return memories

    def delete(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> bool:
        """Delete memories by ID or scope.

        Args:
            memory_id: Specific memory ID to delete.
            user_id: Delete all memories for this user.
            session_id: Delete all memories for this session.
            agent_id: Delete all memories for this agent.

        Returns:
            True if deletion was successful, False otherwise.

        """
        try:
            if memory_id:
                self.memory.delete(memory_id=memory_id)
            else:
                self.memory.delete_all(
                    user_id=user_id,
                    run_id=session_id,
                    agent_id=agent_id,
                )
            logger.info(f"Deleted memories: id={memory_id}, user={user_id}, session={session_id}")
            success = True
        except Exception as e:
            logger.error(f"Error deleting memories: {e}")
            success = False
        return success


# Module-level singleton for convenience
_memory_instance: ConversationalMemory | None = None


def get_memory_service(config: Config | None = None) -> ConversationalMemory:
    """Get or create the memory service singleton.

    Args:
        config: Optional Config instance for first-time initialization.

    Returns:
        The ConversationalMemory singleton instance.

    """
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationalMemory(config)
    return _memory_instance
