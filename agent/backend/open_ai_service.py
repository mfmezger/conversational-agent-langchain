"""This script is used to initialize the Qdrant db backend with Azure OpenAI."""
import os
from typing import Any, List, Tuple

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
            raise ValueError("API Token not provided!")

        self.cfg = cfg

        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = self.cfg.qdrant.collection_name_openai

    def create_collection(self, name: str) -> None:
        """Create a new collection in the Vector Database.

        Args:
            name (str): The name of the new collection.
        """
        raise NotImplementedError

    def get_db_connection(self) -> Qdrant:
        """Initializes a connection to the Qdrant DB.

        Returns:
            Qdrant: An Langchain Instance of the Qdrant DB.
        """
        if self.cfg.openai.azure:
            embedding = AzureOpenAIEmbeddings(deployment=cfg.openai.deployment, openai_api_version="2023-05-15", openai_api_key=open_ai_token)  # type: ignore
        else:
            embedding = OpenAIEmbeddings(model=self.cfg.openai.deployment, openai_api_key=self.open_ai_token)

        if not self.collection_name:
            collection_name = self.cfg.qdrant.collection_name_openai

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
        vector_db: Qdrant = get_db_connection(open_ai_token=self.open_ai_token, collection_name=self.collection_name)

        splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

        loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFium2Loader)
        docs = loader.load_and_split(splitter)

        logger.info(f"Loaded {len(docs)} documents.")
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        vector_db.add_texts(texts=texts, metadatas=metadatas)
        logger.info("SUCCESS: Texts embedded.")

    def search(self, search: SearchRequest) -> List[Tuple[Document, float]]:
        """Searches the documents in the Qdrant DB with a specific query.

        Args:
            open_ai_token (str): The OpenAI API token.
            query (str): The question for which documents should be searched.

        Returns:
            List[Tuple[Document, float]]: A list of search results, where each result is a tuple
            containing a Document object and a float score.
        """
        vector_db = get_db_connection(open_ai_token=self.open_ai_token, collection_name=self.collection_name)

        docs = vector_db.similarity_search_with_score(search.query, k=search.amount, score_threshold=search.filtering.threshold)
        logger.info("SUCCESS: Documents found.")
        return docs

    def summarize_text(self, text: str, token: str) -> str:
        """Summarizes the given text using the OpenAI API.

        Args:
            text (str): The text to be summarized.
            token (str): The token for the OpenAI API.

        Returns:
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

    def send_completion(self, text: str, query: str) -> str:
        """Sent completion request to OpenAI API.

        Args:
            text (str): The text on which the completion should be based.
            query (str): The query for the completion.
            token (str): The token for the OpenAI API.
            cfg (DictConfig):

        Returns:
            str: Response from the OpenAI API.
        """
        prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text, query=query, language="de")

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

    def send_custom_completion_openai(
        self,
        token: str,
        prompt: str,
        model: str = "gpt3.5",
        max_tokens: int = 256,
        stop_sequences: List[str] = ["###"],
        temperature: float = 0,
    ) -> str:
        """Sent completion request to OpenAI API.

        Args:
            text (str): The text on which the completion should be based.
            query (str): The query for the completion.
            token (str): The token for the OpenAI API.
            cfg (DictConfig):

        Returns:
            str: Response from the OpenAI API.
        """
        openai.api_key = self.token
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )

        return response.choices[0].text

    def rag(self, rag_request: RAGRequest) -> tuple[Any, str, dict[Any, Any]]:
        """QA Function for OpenAI LLMs.

        Args:
            token (str): The token for the OpenAI API.
            documents (list[tuple[Document, float]]): The documents to be searched.
            query (str): The question for which the LLM should generate an answer.
            summarization (bool, optional): If the Documents should be summarized. Defaults to False.

        Returns:
            tuple: answer, prompt, meta_data
        """
        # if the list of documents contains only one document extract the text directly
        if len(documents) == 1:
            text = documents[0][0].page_content
            meta_data = documents[0][0].metadata

        else:
            # extract the text from the documents
            texts = [doc[0].page_content for doc in documents]
            if summarization:
                # call summarization
                text = ""
                for t in texts:
                    text += summarize_text_openai(text=t, token=token)

            else:
                # combine the texts to one text
                text = " ".join(texts)
            meta_data = [doc[0].metadata for doc in documents]

        # load the prompt
        prompt = generate_prompt("aleph_alpha_qa.j2", text=text, query=query)

        try:

            # call the luminous api
            answer = send_completion(prompt, token)

        except ValueError as e:
            # if the code is PROMPT_TOO_LONG, split it into chunks
            if e.args[0] == "PROMPT_TOO_LONG":
                logger.info("Prompt too long. Summarizing.")

                # summarize the text
                short_text = summarize_text_openai(text, token)

                # generate the prompt
                prompt = generate_prompt("openai-qa.j2", text=short_text, query=query)

                # call the luminous api
                answer = send_completion(prompt, token)

        # extract the answer
        return answer, prompt, meta_data


if __name__ == "__main__":

    token = os.getenv("OPENAI_API_KEY")

    if not token:
        raise ValueError("OPENAI_API_KEY is not set.")

    openai_service = OpenAIService(collection_name="openai", token=token)

    openai_service.embed_documents(directory="data")

    DOCS = openai_service.search(SearchRequest(query="Was ist Vanille?"))
    print(f"DOCUMENTS: {DOCS}")

    summary = openai_service.rag(RAGRequest(query="Was ist Vanille?", documents=DOCS))

    print(f"SUMMARY: {summary}")
