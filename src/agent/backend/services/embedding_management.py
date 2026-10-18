"""LiteLLM Backend."""

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from agent.utils.embeddings import get_embedding_model
from agent.utils.vdb import VDBResources, generate_collection, init_vdb

load_dotenv()


class EmbeddingManagement:
    """Wrapper for cohere llms."""

    def __init__(self, collection_name: str, resources: VDBResources) -> None:
        """Initialize the embedding service with explicit VDB resources."""
        self.cfg = resources.config
        self.resources = resources
        self.collection_name = collection_name

        embedding = get_embedding_model(self.cfg)

        self.vector_db = init_vdb(
            resources=self.resources,
            collection_name=self.collection_name,
            embedding=embedding,
        )

    def embed_documents(self, directory: str, file_ending: str = ".pdf") -> None:
        """Embeds the documents in the given directory.

        Args:
        ----
            directory (str): PDF Directory.
            file_ending (str): File ending of the documents.

        """
        # TODO: refactor to use markdownit
        if file_ending == ".pdf":
            loader = DirectoryLoader(directory, glob="*" + file_ending, loader_cls=PyPDFium2Loader)
        elif file_ending == ".txt":
            loader = DirectoryLoader(directory, glob="*" + file_ending, loader_cls=TextLoader)
        else:
            msg = "File ending not supported."
            raise ValueError(msg)

        splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200, length_function=len, separators=["\n\n", "\n", ".", "!"])

        docs = loader.load_and_split(splitter)

        logger.info(f"Loaded {len(docs)} documents.")
        text_list = [doc.page_content for doc in docs]
        metadata_list = [doc.metadata for doc in docs]

        for m in metadata_list:
            # only when there are / in the source
            if "/" in m["source"]:
                m["source"] = m["source"].split("/")[-1]

        self.vector_db.add_texts(texts=text_list, metadatas=metadata_list)

        logger.info("SUCCESS: Texts embedded.")

    def create_collection(self, name: str) -> bool:
        """Create a new collection in the Vector Database."""
        generate_collection(
            client=self.resources.sync_client,
            collection_name=name,
            embeddings_size=self.cfg.embedding_size,
        )
        return True


if __name__ == "__main__":
    import asyncio

    from agent.utils.config import Config
    from agent.utils.vdb import create_vdb_resources

    resources = create_vdb_resources(Config())
    try:
        service = EmbeddingManagement(collection_name="default", resources=resources)
        service.embed_documents(directory="tests/resources/")
    finally:
        asyncio.run(resources.close())
