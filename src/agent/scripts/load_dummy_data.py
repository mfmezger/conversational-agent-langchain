"""Simple Upload script for test data."""

from agent.backend.services.embedding_management import EmbeddingManagement
from agent.utils.vdb import generate_collection


def main() -> None:
    """Generating test collection and uploading data for testing."""
    generate_collection(collection_name="asdf", embeddings_size=1536)
    vdb = EmbeddingManagement(collection_name="test")
    vdb.embed_documents(directory="resources")


if __name__ == "__main__":
    main()
