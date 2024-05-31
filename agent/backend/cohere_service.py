"""Cohere Backend."""
import os

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from omegaconf import DictConfig
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader, TextLoader
from langchain.text_splitter import NLTKTextSplitter
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBackend, LLMBase
from loguru import logger
from agent.data_model.request_data_model import (
    Filtering,
    RAGRequest,
    SearchRequest,
)
from agent.utils.vdb import init_vdb


load_dotenv()


class CohereService(LLMBase):

    """Wrapper for cohere llms."""

    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str, token: str) -> None:
        """Init the OpenAI Service."""
        super().__init__(token=token, collection_name=collection_name)

        """Openai Service."""
        if token:
            os.environ["COHERE_API_KEY"] = token

        self.cfg = cfg

        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = self.cfg.qdrant.collection_name_cohere

        embedding = CohereEmbeddings(
            model=self.cfg.cohere_embeddings.embedding_model_name
        )

        self.vector_db = init_vdb(self.cfg, self.collection_name, embedding=embedding)

    def embed_documents(self, directory: str, file_ending: str = ".pdf") -> None:
        """Embeds the documents in the given directory.

        Args:
        ----
            directory (str): PDF Directory.
            file_ending (str): File ending of the documents.

        """
        if file_ending == ".pdf":
            loader = DirectoryLoader(directory, glob="*" + file_ending, loader_cls=PyPDFium2Loader)
        elif file_ending == ".txt":
            loader = DirectoryLoader(directory, glob="*" + file_ending, loader_cls=TextLoader)
        else:
            msg = "File ending not supported."
            raise ValueError(msg)

        splitter = NLTKTextSplitter(length_function=len, chunk_size=500, chunk_overlap=75)

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

    def search(self, search: SearchRequest, filtering: Filtering) -> list:
        """Searches the documents in the Qdrant DB with semantic search."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""

    def rag(self, rag: RAGRequest, search: SearchRequest, filtering: Filtering) -> tuple:
        """Retrieval Augmented Generation."""

    def summarize_text(self, text: str) -> str:
        """Summarize text."""

if __name__ == "__main__":
    cohere_service = CohereService()