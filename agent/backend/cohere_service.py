"""Cohere Backend."""
import os

from dotenv import load_dotenv
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader, TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, chain
from langchain_text_splitters import NLTKTextSplitter
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBase
from agent.data_model.request_data_model import (
    RAGRequest,
    SearchParams,
)
from agent.utils.utility import extract_text_from_langchain_documents, load_prompt_template
from agent.utils.vdb import generate_collection_cohere, init_vdb

load_dotenv()


class CohereService(LLMBase):

    """Wrapper for cohere llms."""

    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str | None, token: str | None) -> None:
        """Init the Cohere Service."""
        super().__init__(token=token, collection_name=collection_name)

        if token:
            os.environ["COHERE_API_KEY"] = token

        self.cfg = cfg

        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = self.cfg.qdrant.collection_name_cohere

        embedding = CohereEmbeddings(model=self.cfg.cohere_embeddings.embedding_model_name)

        template = load_prompt_template(prompt_name="cohere_chat.j2", task="chat")
        self.prompt = ChatPromptTemplate.from_template(template=template, template_format="jinja2")

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
        generate_collection_cohere(self.cfg, name)
        return True

    def search(self, search: SearchParams) -> BaseRetriever:
        """Searches the documents in the Qdrant DB with semantic search."""
        search = dict(search)
        search.pop("query")

        @chain
        def retriever_with_score(query: str) -> list[Document]:
            docs, scores = zip(
                *self.vector_db.similarity_search_with_score(query, k=search["k"], filter=search["filter"], score_threshold=search["score_threshold"]), strict=False
            )
            for doc, score in zip(docs, scores, strict=False):
                doc.metadata["score"] = score

            return docs

        return retriever_with_score

    def rag(self, rag: RAGRequest, search: SearchParams) -> tuple:
        """Retrieval Augmented Generation."""
        search_chain = self.search(search=search)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: extract_text_from_langchain_documents(x["context"]))) | self.prompt | ChatCohere() | StrOutputParser()
        )

        return RunnableParallel({"context": search_chain, "question": RunnablePassthrough()}).assign(answer=rag_chain_from_docs)

    def summarize_text(self, text: str) -> str:
        """Summarize text."""


if __name__ == "__main__":
    query = "Was ist Attention?"

    cohere_service = CohereService(collection_name="", token="")

    # cohere_service.embed_documents(directory="tests/resources/")

    # search_chain = cohere_service.search(search=SearchParams(query=query, amount=3))

    # search_results = search_chain.invoke(query)

    chain = cohere_service.rag(rag=RAGRequest(), search=SearchParams(query=query, amount=3))

    answer = chain.invoke(query)

    logger.info(answer)
