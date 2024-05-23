"""GPT4ALL Backend Service."""

from dotenv import load_dotenv
from gpt4all import GPT4All
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader, TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBase
from agent.data_model.internal_model import RetrievalResults
from agent.data_model.request_data_model import (
    Filtering,
    RAGRequest,
    SearchRequest,
)
from agent.utils.utility import convert_qdrant_result_to_retrieval_results, generate_prompt
from agent.utils.vdb import generate_collection_gpt4all, init_vdb

# nltk.download("punkt")

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

        embedding = GPT4AllEmbeddings()
        self.vector_db = init_vdb(cfg=self.cfg, collection_name=collection_name, embedding=embedding)

        # create retriever from the vector database

        # query = "Was ist Attention?"
        # results = retriever.invoke(query=query)
        # print(results)
        # self.chain =

    def create_search_chain(self, search_kwargs: dict[str, any] | None = None):
        if search_kwargs is None:
            search_kwargs = {}
        return self.vector_db.as_retriever(search_kwargs=search_kwargs)

    def create_rag_chain(self, search_chain):
        llm = GPT4All(self.cfg.gpt4all_completion.completion_model)
        return search_chain | llm

    def create_collection(self, name: str) -> bool:
        """Create a new collection in the Vector Database."""
        generate_collection_gpt4all(self.vector_db.client, name)
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

    def search(self, search: SearchRequest, filtering: Filtering) -> list[RetrievalResults]:
        """Searches the documents in the Qdrant DB with a specific query.

        Args:
        ----
            search (SearchRequest): The search request.
            filtering (Filtering): The filtering parameters.

        Returns:
        -------
            List[Tuple[Document, float]]: A list of search results, where each result is a tuple
            containing a Document object and a float score.

        """
        docs = self.vector_db.similarity_search_with_score(query=search.query, k=search.amount, score_threshold=filtering.threshold, filter=filtering.filter)
        logger.info(f"SUCCESS: {len(docs)} Documents found.")

        return convert_qdrant_result_to_retrieval_results(docs)

    def rag(self, rag_request: RAGRequest, search: SearchRequest, filtering: Filtering) -> tuple:
        """RAG takes a Rag Request Object and performs a semantic search and then a generation.

        Args:
        ----
            rag_request (RAGRequest): The RAG Request Object.
            search (SearchRequest): The search request.
            filtering (Filtering): The filtering parameters.

        Returns:
        -------
            Tuple[str, str, List[RetrievalResults]]: The answer, the prompt and the metadata.

        """
        documents = self.search(search=search, filtering=filtering)
        if search.amount == 0:
            msg = "No documents found."
            raise ValueError(msg)
        text = "\n".join([doc.document for doc in documents]) if search.amount > 1 else documents[0].document

        # TODO: Add the history to the prompt
        prompt = generate_prompt(prompt_name="gpt4all-completion.j2", text=text, query=search.query, language=rag_request.language)

        answer = self.generate(prompt)

        return answer, prompt, documents


if __name__ == "__main__":
    query = "Was ist Attention?"

    gpt4all_service = GPT4AllService(collection_name="gpt4all", token="")

    # gpt4all_service.embed_documents(directory="tests/resources/")

    retriever = gpt4all_service.create_search_chain(search_kwargs={"k": 3})

    results = (retriever.invoke(query),)  # config={'callbacks': [ConsoleCallbackHandler()]})

    rag_chain = gpt4all_service.create_rag_chain(search_chain=retriever)

    # docs = gpt4all_service.search(SearchRequest(query, amount=3), Filtering(threshold=0.0, collection_name="gpt4all"))

    # logger.info(f"Documents: {docs}")

    # answer, prompt, meta_data = gpt4all_service.rag(
    #     RAGRequest(language="detect", history={}),
    #     SearchRequest(
    #         query=query,
    #         amount=3,
    #     ),
    #     Filtering(threshold=0.0, collection_name="gpt4all"),
    # )

    # logger.info(f"Answer: {answer}")
    # logger.info(f"Prompt: {prompt}")
    # logger.info(f"Metadata: {meta_data}")
