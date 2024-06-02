"""GPT4ALL Backend Service."""

from dotenv import load_dotenv
from gpt4all import GPT4All
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader, TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, chain
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBase
from agent.data_model.request_data_model import (
    RAGRequest,
    SearchParams,
)
from agent.utils.utility import extract_text_from_langchain_documents, generate_prompt, load_prompt_template
from agent.utils.vdb import generate_collection, init_vdb

load_dotenv()


class GPT4AllService(LLMBase):

    """GPT4ALL Backend Service."""

    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str, token: str | None) -> None:
        """Init the GPT4ALL Service."""
        self.cfg = cfg
        self.token = token

        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = self.cfg.qdrant.collection_name_gpt4all

        embedding = GPT4AllEmbeddings(model_name="nomic-embed-text-v1.5.f16.gguf")

        template = load_prompt_template(prompt_name="cohere_chat.j2", task="chat")
        self.prompt = ChatPromptTemplate.from_template(template=template, template_format="jinja2")

        self.vector_db = init_vdb(cfg=self.cfg, collection_name=collection_name, embedding=embedding)

    def create_collection(self, name: str) -> bool:
        """Create a new collection in the Vector Database."""
        generate_collection(name, self.cfg.gpt4all_embeddings.size)
        return True

    def embed_documents(self, directory: str, file_ending: str = ".pdf") -> None:
        """Embeds the documents in the given directory.

        Args:
        ----
            directory (str): PDF Directory.
            file_ending (str): The file ending of the documents.

        Returns:
        -------
            None

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

    def summarize_text(self, text: str) -> str:
        """Summarize text with GPT4ALL.

        Args:
        ----
            text (str): The text to be summarized.

        Returns:
        -------
            str: The summarized text.

        """
        prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text)

        model = GPT4All(self.cfg.gpt4all_completion.completion_model)

        return model.generate(prompt, max_tokens=300)

    def generate(self, prompt: str) -> str:
        """Complete text with GPT4ALL.

        Args:
        ----
            prompt (str): The prompt to be completed.

        Returns:
        -------
            str: The completed text.

        """
        model = GPT4All(self.cfg.gpt4all_completion.completion_model)

        return model.generate(prompt, max_tokens=250)

    def create_search_chain(self, search: SearchParams) -> BaseRetriever:
        """Searches the documents in the Qdrant DB with semantic search."""

        @chain
        def retriever_with_score(query: str) -> list[Document]:
            docs, scores = zip(
                *self.vector_db.similarity_search_with_score(query, k=search.k, filter=search.filter, score_threshold=search.score_threshold), strict=False
            )
            for doc, score in zip(docs, scores, strict=False):
                doc.metadata["score"] = score

            return docs

        return retriever_with_score

    def create_rag_chain(self, rag: RAGRequest, search: SearchParams) -> tuple:
        """Retrieval Augmented Generation."""
        search_chain = self.create_search_chain(search=search)
        llm = GPT4All(self.cfg.gpt4all_completion.completion_model)

        rag_chain_from_docs = RunnablePassthrough.assign(context=(lambda x: extract_text_from_langchain_documents(x["context"]))) | self.prompt | llm | StrOutputParser()

        return RunnableParallel({"context": search_chain, "question": RunnablePassthrough()}).assign(answer=rag_chain_from_docs)


if __name__ == "__main__":
    query = "Was ist Attention?"

    gpt4all_service = GPT4AllService(collection_name="", token="")

    gpt4all_service.embed_documents(directory="tests/resources/")

    chain = gpt4all_service.create_rag_chain(rag=RAGRequest(), search=SearchParams(query=query, amount=3))

    answer = chain.invoke(query)

    logger.info(answer)
