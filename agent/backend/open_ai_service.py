"""This script is used to initialize the Qdrant db backend with Azure OpenAI."""
import os
from typing import Any

import openai
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader
from langchain_community.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBase
from agent.data_model.request_data_model import RAGRequest, SearchRequest
from agent.utils.utility import generate_prompt
from agent.utils.vdb import init_vdb

load_dotenv()


class OpenAIService(LLMBase):

    """OpenAI Backend Service."""

    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str, token: str) -> None:
        """Init the OpenAI Service."""
        super().__init__(token=token, collection_name=collection_name)

        """Openai Service."""
        if token:
            self.openai_token = token
        else:
            self.openai_token = os.getenv("ALEPH_ALPHA_API_KEY")

        if not self.openai_token:
            msg = "API Token not provided!"
            raise ValueError(msg)

        self.cfg = cfg

        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = self.cfg.qdrant.collection_name_openai

        self.vector_db = self.get_db_connection()

    def create_collection(self, name: str) -> None:
        """Create a new collection in the Vector Database.

        Args:
        ----
            name (str): The name of the new collection.
        """
        raise NotImplementedError

    def get_db_connection(self) -> Qdrant:
        """Initializes a connection to the Qdrant DB.

        Returns
        -------
            Qdrant: An Langchain Instance of the Qdrant DB.
        """
        if self.cfg.openai.azure:
            embedding = AzureOpenAIEmbeddings(deployment=cfg.openai.deployment, openai_api_version="2023-05-15", openai_api_key=open_ai_token)  # type: ignore
        else:
            embedding = OpenAIEmbeddings(model=self.cfg.openai.deployment, openai_api_key=self.open_ai_token)

        return init_vdb(self.cfg, self.collection_name, embedding)

    def embed_documents(self, directory: str) -> None:
        """embedd_documents embedds the documents in the given directory.

        :param cfg: Configuration from the file
        :type cfg: DictConfig
        :param dir: PDF Directory
        :type dir: str
        :param open_ai_token: OpenAI API Token
        :type open_ai_token: str
        """
        if file_ending == "*.pdf":
            loader = DirectoryLoader(directory, glob=file_ending, loader_cls=PyPDFium2Loader)
        elif file_ending == "*.txt":
            loader = DirectoryLoader(directory, glob=file_ending, loader_cls=TextLoader)
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

    def search(self, search: SearchRequest) -> list[tuple[Document, float]]:
        """Searches the documents in the Qdrant DB with a specific query.

        Args:
        ----
            open_ai_token (str): The OpenAI API token.
            query (str): The question for which documents should be searched.

        Returns:
        -------
            List[Tuple[Document, float]]: A list of search results, where each result is a tuple
            containing a Document object and a float score.
        """
        docs = self.vector_db.similarity_search_with_score(search.query, k=search.amount, score_threshold=search.filtering.threshold)
        logger.info("SUCCESS: Documents found.")
        return docs

    def summarize_text(self, text: str, token: str) -> str:
        """Summarizes the given text using the OpenAI API.

        Args:
        ----
            text (str): The text to be summarized.
            token (str): The token for the OpenAI API.

        Returns:
        -------
            str: The summary of the text.
        """
        prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text, language="de")

        openai.api_key = token
        response = openai.Completion.create(
            engine=self.cfg.openai.model,
            prompt=prompt,
            temperature=self.cfg.openai.temperature,
            max_tokens=self.cfg.openai.max_tokens,
            top_p=self.cfg.openai.top_p,
            frequency_penalty=self.cfg.openai.frequency_penalty,
            presence_penalty=self.cfg.openai.presence_penalty,
            best_of=self.cfg.openai.best_of,
            stop=self.cfg.openai.stop,
        )

        return response.choices[0].text

    def generate(self, prompt: str) -> str:
        """Sent completion request to OpenAI API.

        Args:
        ----
            prompt (str): The text on which the completion should be based.

        Returns:
        -------
            str: Response from the OpenAI API.
        """
        openai.api_key = token
        response = openai.Completion.create(
            engine=self.cfg.openai.model,
            prompt=prompt,
            temperature=self.cfg.openai.temperature,
            max_tokens=self.cfg.openai.max_tokens,
            top_p=self.cfg.openai.top_p,
            frequency_penalty=self.cfg.openai.frequency_penalty,
            presence_penalty=self.cfg.openai.presence_penalty,
            best_of=self.cfg.openai.best_of,
            stop=self.cfg.openai.stop,
        )

        return response.choices[0].text

    def rag(self, rag_request: RAGRequest) -> tuple[Any, str, dict[Any, Any]]:
        """QA Function for OpenAI LLMs.

        Args:
        ----
            rag_request (RAGRequest): The RAG Request Object.

        Returns:
        -------
            tuple: answer, prompt, meta_data
        """
        documents = self.search(rag_request.search)
        if rag_request.search.amount == 0:
            msg = "No documents found."
            raise ValueError(msg)
        if rag_request.search.amount > 1:
            # extract all documents
            text = "\n".join([doc.document for doc in documents])
        else:
            text = documents[0].document

        prompt = generate_prompt(prompt_name="openai-qa.j2", text=text, query=rag_request.search.query)

        answer = self.generate(prompt)

        return answer, prompt, documents


if __name__ == "__main__":
    token = os.getenv("AZURE_OPENAI_API_KEY")

    if not token:
        msg = "OPENAI_API_KEY is not set."
        raise ValueError(msg)

    openai_service = OpenAIService(collection_name="openai", token="")

    openai_service.embed_documents(directory="tests/resources/")

    docs = openai_service.search(
        SearchRequest(
            query="Was ist Attention?",
            amount=3,
            filtering=Filtering(threshold=0.0, collection_name="openai"),
            llm_backend=LLMBackend(token="gpt4all", provider=LLMProvider.GPT4ALL),
        )
    )

    logger.info(f"Documents: {docs}")

    answer, prompt, meta_data = openai_service.rag(
        RAGRequest(
            search=SearchRequest(
                query="Was ist Attention?",
                amount=3,
                filtering=Filtering(threshold=0.0, collection_name="gpt4all"),
                llm_backend=LLMBackend(token="gpt4all", provider=LLMProvider.GPT4ALL),
            ),
            documents=docs,
            query="Was ist das?",
        )
    )

    logger.info(f"Answer: {answer}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Metadata: {meta_data}")
