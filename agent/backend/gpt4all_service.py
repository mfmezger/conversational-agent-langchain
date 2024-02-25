"""GPT4ALL Backend Service."""
from dotenv import load_dotenv
from gpt4all import GPT4All
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBase
from agent.data_model.internal_model import RetrievalResults
from agent.data_model.request_data_model import (
    Filtering,
    LLMBackend,
    RAGRequest,
    SearchRequest,
)
from agent.utils.utility import (
    convert_qdrant_result_to_retrieval_results,
    generate_prompt,
)
from agent.utils.vdb import init_vdb

load_dotenv()


class GPT4AllService(LLMBase):
    """GPT4ALL Backend Service."""

    @load_config(location="config/db.yml")
    def __init__(self, cfg: DictConfig, collection_name: str, token: str | None) -> None:
        """Init the GPT4ALL Service."""
        self.cfg = cfg

        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = self.cfg.qdrant.collection_name_gpt4all

        self.vector_db = self.get_db_connection(self.collection_name)

    def get_db_connection(self, collection_name: str) -> Qdrant:
        """Initializes a connection to the Qdrant DB.

        Args:
            cfg (DictConfig): The configuration file loaded via OmegaConf.
            aleph_alpha_token (str): The Aleph Alpha API token.

        Returns:
            Qdrant: The Qdrant DB connection.
        """
        embedding = GPT4AllEmbeddings()

        return init_vdb(self.cfg, self.collection_name, embedding)

    def create_collection(self, name: str):
        """Create a new collection in the Vector Database."""
        pass

    def embed_documents(self, directory: str, file_ending: str = "*.pdf") -> None:
        """Embeds the documents in the given directory.

        Args:
            cfg (DictConfig): Configuration from the file.
            dir (str): PDF Directory.

        Returns:
            None
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

    def summarize_text(self, text: str) -> str:
        """Summarize text with GPT4ALL.

        Args:
            text (str): The text to be summarized.

        Returns:
            str: The summarized text.
        """
        prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text)

        model = GPT4All(self.cfg.gpt4all_completion.completion_model)

        return model.generate(prompt, max_tokens=300)

    def generate(self, prompt: str) -> str:
        """Complete text with GPT4ALL.

        Args:
            text (str): The text as basic input.
            query (str): The query to be inserted into the template.

        Returns:
            str: The completed text.
        """
        model = GPT4All(self.cfg.gpt4all_completion.completion_model)

        return model.generate(prompt, max_tokens=250)

    def search(self, search: SearchRequest) -> list[RetrievalResults]:
        """Searches the documents in the Qdrant DB with a specific query.

        Args:
            open_ai_token (str): The OpenAI API token.
            query (str): The question for which documents should be searched.

        Returns:
            List[Tuple[Document, float]]: A list of search results, where each result is a tuple
            containing a Document object and a float score.
        """
        docs = self.vector_db.similarity_search_with_score(query=search.query, k=search.amount, score_threshold=search.filtering.threshold)
        logger.info(f"SUCCESS: {len(docs)} Documents found.")

        return convert_qdrant_result_to_retrieval_results(docs)

    def rag(self, rag_request: RAGRequest) -> tuple:
        """RAG takes a Rag Request Object and performs a semantic search and then a generation.

        Args:
            rag_request (RAGRequest): The RAG Request Object.

        Returns:
            Tuple[str, str, List[RetrievalResults]]: The answer, the prompt and the metadata.
        """
        documents = self.search(rag_request.search)
        if rag_request.search.amount == 0:
            raise ValueError("No documents found.")
        if rag_request.search.amount > 1:
            # extract all documents
            text = "\n".join([doc.document for doc in documents])
        else:
            text = documents[0].document

        prompt = generate_prompt(prompt_name="gpt4all-completion.j2", text=text, query=rag_request.search.query)

        answer = self.completion_text_gpt4all(prompt)

        return answer, prompt, documents


if __name__ == "__main__":

    gpt4all_service = GPT4AllService(collection_name="gpt4all", token="")

    gpt4all_service.embed_documents(directory="tests/resources/")

    docs = gpt4all_service.search(
        SearchRequest(
            query="Was ist Attention?",
            amount=3,
            filtering=Filtering(threshold=0.0, collection_name="gpt4all"),
            llm_backend=LLMBackend(token="gpt4all, provider=LLMProvider.GPT4ALL"),
        )
    )

    logger.info(f"Documents: {docs}")

    answer, prompt, meta_data = gpt4all_service.rag(
        RAGRequest(
            search=SearchRequest(
                query="Was ist Attention?",
                amount=3,
                filtering=Filtering(threshold=0.0, collection_name="gpt4all"),
                llm_backend=LLMBackend(token="gpt4all, provider=LLMProvider.GPT4ALL"),
            ),
            documents=docs,
            query="Was ist das?",
        )
    )

    logger.info(f"Answer: {answer}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Metadata: {meta_data}")
