"""Script is used to initialize the Qdrant db backend with (Azure) OpenAI."""
import os

import openai
from dotenv import load_dotenv
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import chain
from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBase
from agent.data_model.request_data_model import RAGRequest, SearchParams
from agent.utils.utility import generate_prompt, load_prompt_template
from agent.utils.vdb import generate_collection, init_vdb

load_dotenv()


class OpenAIService(LLMBase):

    """OpenAI Backend Service."""

    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str) -> None:
        """Init the OpenAI Service."""
        super().__init__(collection_name=collection_name)

        """Openai Service."""
        self.cfg = cfg

        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = self.cfg.qdrant.collection_name_openai

        template = load_prompt_template(prompt_name="cohere_chat.j2", task="chat")
        self.prompt = ChatPromptTemplate.from_template(template=template, template_format="jinja2")

        if self.cfg.openai_embeddings.azure:
            embedding = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment=self.cfg.openai_embeddings.embedding_model_name,
                openai_api_version=self.cfg.openai_embeddings.openai_api_version,
            )
        else:
            embedding = OpenAIEmbeddings(model=self.cfg.openai_embeddings.embedding_model_name)

        self.vector_db = init_vdb(self.cfg, self.collection_name, embedding=embedding)

    def create_collection(self, name: str) -> bool:
        """Create a new collection in the Vector Database.

        Args:
        ----
            name (str): The name of the new collection.

        """
        generate_collection(name, self.cfg.openai_embeddings.size)
        logger.info(f"SUCCESS: Collection {name} created.")
        return True

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


    def summarize_text(self, text: str) -> str:
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

        openai.api_key = self.token
        response = openai.chat.completions.create(
            model=self.cfg.openai_completion.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.openai_completion.temperature,
            max_tokens=self.cfg.openai_completion.max_tokens,
            top_p=self.cfg.openai_completion.top_p,
            frequency_penalty=self.cfg.openai_completion.frequency_penalty,
            presence_penalty=self.cfg.openai_completion.presence_penalty,
            stop=self.cfg.openai_completion.stop,
            stream=False,
        )

        return response.choices[0].messages.content


if __name__ == "__main__":
    query = "Was ist Attention?"

    openai_service = OpenAIService(collection_name="", token="")

    openai_service.embed_documents(directory="tests/resources/")

    chain = openai_service.create_rag_chain(rag=RAGRequest(), search=SearchParams(query=query, amount=3))

    answer = chain.invoke(query)

    logger.info(answer)
