"""Simple upload script for test data."""

import asyncio

from agent.backend.services.embedding_management import EmbeddingManagement
from agent.utils.config import Config
from agent.utils.vdb import create_vdb_resources


def main() -> None:
    """Generate a test collection and upload data for testing."""
    resources = create_vdb_resources(Config())
    try:
        service = EmbeddingManagement(collection_name="default", resources=resources)
        service.embed_documents(directory="resources")
    finally:
        asyncio.run(resources.close())


if __name__ == "__main__":
    main()
