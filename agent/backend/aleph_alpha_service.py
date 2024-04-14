"""The script to initialize the Qdrant db backend with aleph alpha."""

import os

import nltk
import numpy as np
from aleph_alpha_client import (
    Client,
    CompletionRequest,
    Document,
    ExplanationRequest,
    Prompt,
    SummarizationRequest,
)
from dotenv import load_dotenv
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader, TextLoader
from langchain_community.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain_community.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBase
from agent.data_model.request_data_model import (
    Filtering,
    RAGRequest,
    SearchRequest,
)
from agent.utils.utility import (
    convert_qdrant_result_to_retrieval_results,
    generate_collection_aleph_alpha,
    generate_prompt,
)
from agent.utils.vdb import init_vdb

nltk.download("punkt")  # This needs to be installed for the tokenizer to work.
load_dotenv()

aleph_alpha_token = os.getenv("ALEPH_ALPHA_API_KEY")
tokenizer = None


class AlephAlphaService(LLMBase):

    """Aleph Alpha Strategy implementation."""

    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str, token: str) -> None:
        """Initialize the Aleph Alpha Service."""
        super().__init__(token=token, collection_name=collection_name)
        """Initialize the Aleph Alpha Service."""
        if token:
            self.aleph_alpha_token = token
        else:
            self.aleph_alpha_token = os.getenv("ALEPH_ALPHA_API_KEY")

        if not self.aleph_alpha_token:
            msg = "API Token not provided!"
            raise ValueError(msg)

        self.cfg = cfg

        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = self.cfg.qdrant.collection_name_aa

        self.vector_db = self.get_db_connection(self.collection_name)

    def get_tokenizer(self) -> None:
        """Initialize the tokenizer."""
        client = Client(token=self.aleph_alpha_token)
        self.tokenizer = client.tokenizer("luminous-base")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text.

        Args:
        ----
            text (str): The text to count the tokens for.

        Returns:
        -------
            int: Number of tokens.
        """
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def get_db_connection(self, collection_name: str) -> Qdrant:
        """Initializes a connection to the Qdrant DB.

        Args:
        ----
            collection_name (str): The name of the collection in the Qdrant DB.

        Returns:
        -------
            Qdrant: The Qdrant DB connection.
        """
        embedding = AlephAlphaAsymmetricSemanticEmbedding(
            model=self.cfg.aleph_alpha_embeddings.model_name,
            aleph_alpha_api_key=self.aleph_alpha_token,
            normalize=self.cfg.aleph_alpha_embeddings.normalize,
            compress_to_size=self.cfg.aleph_alpha_embeddings.compress_to_size,
        )

        return init_vdb(self.cfg, collection_name, embedding)

    def create_collection(self, name: str) -> bool:
        """Create a new collection in the Qdrant DB.

        Args:
        ----
            name (str): The name of the new collection.

        """
        generate_collection_aleph_alpha(self.vector_db.client, name, self.cfg.aleph_alpha_embeddings.size)
        return True

    def summarize_text(self, text: str) -> str:
        """Summarizes the given text using the Luminous API.

        Args:
        ----
            text (str): The text to be summarized.
            token (str): The token for the Luminous API.

        Returns:
        -------
            str: The summary of the text.
        """
        # TODO: rewrite because deprecated.
        client = Client(token=self.aleph_alpha_token)
        document = Document.from_text(text=text)
        request = SummarizationRequest(document=document)
        response = client.summarize(request=request)

        return response.summary

    def generate(self, text: str) -> str:
        """Sends a completion request to the Luminous API.

        Args:
        ----
            text (str): The prompt to be sent to the API.
            token (str): The token for the Luminous API.

        Returns:
        -------
            str: The response from the API.

        Raises:
        ------
            ValueError: If the text or token is None or empty, or if the response or completion is empty.
        """
        if not text:
            msg = "Text cannot be None or empty."
            raise ValueError(msg)

        client = Client(token=self.aleph_alpha_token)

        request = CompletionRequest(
            prompt=Prompt.from_text(text),
            maximum_tokens=self.cfg.aleph_alpha_completion.max_tokens,
            stop_sequences=[self.cfg.aleph_alpha_completion.stop_sequences],
            repetition_penalties_include_completion=self.cfg.aleph_alpha_completion.repetition_penalties_include_completion,
        )
        response = client.complete(request, model=self.cfg.aleph_alpha_completion.model)

        # ensure that the response is not empty
        if not response.completions:
            msg = "Response is empty."
            raise ValueError(msg)

        # ensure that the completion is not empty
        if not response.completions[0].completion:
            msg = "Completion is empty."
            raise ValueError(msg)

        return str(response.completions[0].completion)

    def embed_documents(self, directory: str, file_ending: str = ".pdf") -> None:
        """Embeds the documents in the given directory in the Aleph Alpha database.

        This method uses the Directory Loader for PDFs and the PyPDFium2Loader to load the documents.
        The documents are then added to the Qdrant DB which embeds them without deleting the old collection.

        Args:
        ----
            directory (str): The directory containing the PDFs to embed.
            aleph_alpha_token (str): The Aleph Alpha API token.
            file_ending (str): The file ending of the documents to embed.

        Returns:
        -------
            None
        """
        if file_ending == ".pdf":
            loader = DirectoryLoader(directory, glob=file_ending, loader_cls=PyPDFium2Loader)
        elif file_ending == ".txt":
            loader = DirectoryLoader(directory, glob=file_ending, loader_cls=TextLoader)
        else:
            msg = "File ending not supported."
            raise ValueError(msg)

        self.get_tokenizer()

        splitter = NLTKTextSplitter(length_function=self.count_tokens, chunk_size=300, chunk_overlap=50)
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

    def search(self, search: SearchRequest, filtering: Filtering) -> list[tuple[LangchainDocument, float]]:
        """Searches the Aleph Alpha service for similar documents.

        Args:
        ----
            search (SearchRequest): Search  Request
            llm_backend (LLMBackend): The LLM Backend.
            filtering (Filtering): The filtering object.

        Returns:
        -------
            List[Tuple[Document, float]]: A list of tuples containing the documents and their similarity scores.
        """
        docs = self.vector_db.similarity_search_with_score(query=search.query, k=search.amount, score_threshold=filtering.threshold)
        logger.info(f"SUCCESS: {len(docs)} Documents found.")

        return convert_qdrant_result_to_retrieval_results(docs)

    def rag(self, rag: RAGRequest, search: SearchRequest, filtering: Filtering) -> tuple:
        """QA takes a list of documents and returns a list of answers.

        Args:
        ----
            rag (RAGRequest): The request for the RAG endpoint.
            search (SearchRequest): The search request object.
            filtering (Filtering): The filtering object.

        Returns:
        -------
            Tuple[str, str, List[RetrievalResults]]: The answer, the prompt and the metadata.
        """
        documents = self.search(search=search, filtering=filtering)
        if search.amount == 0:
            msg = "No documents found."
            raise ValueError(msg)
        text = "\n".join([doc.document for doc in documents]) if len(documents) > 1 else documents[0].document

        prompt = generate_prompt(prompt_name="aleph_alpha_qa.j2", text=text, query=search.query, language=rag.language)

        answer = self.generate(prompt)

        return answer, prompt, documents

    def explain_qa(self, document: LangchainDocument, explain_threshold: float, query: str) -> tuple:
        """Explian QA WIP."""
        text = document[0][0].page_content
        meta_data = document[0][0].metadata

        # load the prompt
        prompt = generate_prompt("aleph_alpha_qa.j2", text=text, query=query)

        answer = self.send_completion_request(prompt, self.aleph_alpha_token)

        exp_req = ExplanationRequest(Prompt.from_text(prompt), answer, control_factor=0.1, prompt_granularity="sentence", normalize=True)
        client = Client(token=self.aleph_alpha_token)

        response_explain = client.explain(exp_req, model=self.cfg.aleph_alpha_completion.model)
        explanations = response_explain.explanations[0].items[0].scores

        # if all of the scores are belo 0.7 raise an error
        if all(item.score < explain_threshold for item in explanations):
            msg = "All scores are below explain_threshold."
            raise ValueError(msg)

        # remove element if the text contains Response: or Instructions:
        for exp in explanations:
            txt = prompt[exp.start : exp.start + exp.length]
            if "Response:" in txt or "Instruction:" in txt:
                explanations.remove(exp)

        # pick the top explanation based on score
        top_explanation = max(explanations, key=lambda x: x.score)

        # get the start and end of the explanation
        start = top_explanation.start
        end = top_explanation.start + top_explanation.length

        # get the explanation from the prompt
        explanation = prompt[start:end]

        # get the score
        score = np.round(top_explanation.score, decimals=3)

        # get the text from the document
        text = document[0][0].page_content

        return explanation, score, text, answer, meta_data

    def process_documents_aleph_alpha(self, folder: str, processing_type: str) -> list[str]:
        """Process the documents in the given folder.

        Args:
        ----
            folder (str): Folder where the documents are located.
            token (str): The Aleph Alpha API Token.
            processing_type (str): The processing_type of the documents.

        Raises:
        ------
            ValueError: If the type is not one of 'qa', 'summarization', or 'invoice'.
        """
        # load the documents
        loader = DirectoryLoader(folder, glob="*.pdf", loader_cls=PyPDFium2Loader)

        # load the documents
        docs = loader.load()

        # load the correct prompt
        match processing_type:
            case "qa":
                raise NotImplementedError
            case "summarization":
                raise NotImplementedError
            case "invoice":
                # load the prompt
                prompt_name = "aleph-alpha-invoice.j2"
            case _:
                msg = "Type must be one of 'qa', 'summarization', or 'invoice'."
                raise ValueError(msg)

        # generate the prompt
        answers = []
        # iterate over the documents
        for doc in docs:
            # combine the prompt and the text
            prompt_text = generate_prompt(prompt_name=prompt_name, text=doc.page_content, language="en")
            # call the luminous api
            answer = self.send_completion_request(prompt_text)

            answers.append(answer)

        return answers


if __name__ == "__main__":
    token = os.getenv("ALEPH_ALPHA_API_KEY")

    if not token:
        msg = "Token cannot be None or empty."
        raise ValueError(msg)

    aa_service = AlephAlphaService(token=token, collection_name="aleph_alpha")

    aa_service.embed_documents("tests/resources/")
    # open the text file and read the text
    docs = aa_service.search(SearchRequest(query="Was ist Attention?", amount=3), Filtering(threshold=0.0, collection_name="aleph_alpha"))

    logger.info(f"Documents: {docs}")

    answer, prompt, meta_data = aa_service.rag(
        RAGRequest(language="detect", history={}),
        SearchRequest(
            query="Was ist Attention?",
            amount=3,
        ),
        Filtering(threshold=0.0, collection_name="aleph_alpha"),
    )

    logger.info(f"Answer: {answer}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Metadata: {meta_data}")
