"""Cohere Backend."""

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader, TextLoader
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import chain
from langchain_text_splitters import NLTKTextSplitter
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBase
from agent.data_model.request_data_model import (
    SearchParams,
)
from agent.utils.vdb import generate_collection, init_vdb

load_dotenv()


class CohereService(LLMBase):
    """Wrapper for cohere llms."""

    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str | None) -> None:
        """Init the Cohere Service."""
        super().__init__(collection_name=collection_name)

        self.cfg = cfg

        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = self.cfg.qdrant.collection_name_cohere

        embedding = CohereEmbeddings(model=self.cfg.cohere_embeddings.embedding_model_name)

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
        generate_collection(name, self.cfg.cohere_embeddings.size)
        return True

    def create_search_chain(self, search: SearchParams) -> BaseRetriever:
        """Searches the documents in the Qdrant DB with semantic search."""

        @chain
        def retriever_with_score(query: str) -> list[Document]:
            """Defines a retriever that returns the score.

            Args:
            ----
                query (str): Query the user asks.

            Returns:
            -------
                list[Document]: List of Langchain Documents.

            """
            docs, scores = zip(
                *self.vector_db.similarity_search_with_score(query, k=search.k, filter=search.filter, score_threshold=search.score_threshold), strict=False
            )
            for doc, score in zip(docs, scores, strict=False):
                doc.metadata["score"] = score

            return docs

        return retriever_with_score

    def summarize_text(self, text: str) -> str:
        """Summarize text."""


if __name__ == "__main__":
    query = "Was ist Attention?"

    cohere_service = CohereService(collection_name="")

    cohere_service.embed_documents(directory="tests/resources/")

    chain = cohere_service.create_search_chain(SearchParams(query=query, amount=3))

    answer = chain.invoke(query)

    logger.info(answer)
