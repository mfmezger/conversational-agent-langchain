"""The script to initialize the Qdrant db backend with aleph alpha."""

import os
from typing import List, Tuple

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
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader
from langchain_community.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain_community.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBase
from agent.data_model.request_data_model import RAGRequest, SearchRequest
from agent.utils.utility import generate_prompt
from agent.utils.vdb import init_vdb

nltk.download("punkt")  # This needs to be installed for the tokenizer to work.
load_dotenv()

aleph_alpha_token = os.getenv("ALEPH_ALPHA_API_KEY")
tokenizer = None


class AlephAlphaService(LLMBase):
    """Aleph Alpha Strategy implementation."""

    @load_config(location="config/db.yml")
    def __init__(self, cfg: DictConfig, collection_name: str, token: str) -> None:
        """Initialize the Aleph Alpha Service."""
        super().__init__(token=token, collection_name=collection_name)
        """Initialize the Aleph Alpha Service."""
        if token:
            self.aleph_alpha_token = token
        else:
            self.aleph_alpha_token = os.getenv("ALEPH_ALPHA_API_KEY")

        if not self.aleph_alpha_token:
            raise ValueError("API Token not provided!")

        self.cfg = cfg

        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = self.cfg.qdrant.collection_name_aa

    def get_tokenizer(self):
        """Initialize the tokenizer."""
        client = Client(token=self.aleph_alpha_token)
        self.tokenizer = client.tokenizer("luminous-base")

    def count_tokens(self, text: str):
        """Count the number of tokens in the text.

        Args:
            text (str): The text to count the tokens for.

        Returns:
            int: Number of tokens.
        """
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def get_db_connection(self) -> Qdrant:
        """Initializes a connection to the Qdrant DB.

        Args:
            cfg (DictConfig): The configuration file loaded via OmegaConf.
            aleph_alpha_token (str): The Aleph Alpha API token.

        Returns:
            Qdrant: The Qdrant DB connection.
        """
        embedding = AlephAlphaAsymmetricSemanticEmbedding(
            model=self.cfg.aleph_alpha_embeddings.model_name,
            aleph_alpha_api_key=self.aleph_alpha_token,
            normalize=self.cfg.aleph_alpha_embeddings.normalize,
            compress_to_size=self.cfg.aleph_alpha_embeddings.compress_to_size,
        )

        return init_vdb(self.cfg, collection_name, embedding)

    def create_collection(self, name: str):
        """Create a new collection in the Qdrant DB."""
        pass

    def summarize_text(self, text: str) -> str:
        """Summarizes the given text using the Luminous API.

        Args:
            text (str): The text to be summarized.
            token (str): The token for the Luminous API.

        Returns:
            str: The summary of the text.
        """
        # TODO: rewrite because deprecated.
        client = Client(token=token)
        document = Document.from_text(text=text)
        request = SummarizationRequest(document=document)
        response = client.summarize(request=request)

        return response.summary

    def send_completion_request(self, text: str) -> str:
        """Sends a completion request to the Luminous API.

        Args:
            text (str): The prompt to be sent to the API.
            token (str): The token for the Luminous API.

        Returns:
            str: The response from the API.

        Raises:
            ValueError: If the text or token is None or empty, or if the response or completion is empty.
        """
        if not text:
            raise ValueError("Text cannot be None or empty.")

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
            raise ValueError("Response is empty.")

        # ensure that the completion is not empty
        if not response.completions[0].completion:
            raise ValueError("Completion is empty.")

        return str(response.completions[0].completion)

    def embed_documents(self, directory: str, file_ending: str = "*.pdf"):
        """Embeds the documents in the given directory in the Aleph Alpha database.

        This method uses the Directory Loader for PDFs and the PyPDFium2Loader to load the documents.
        The documents are then added to the Qdrant DB which embeds them without deleting the old collection.

        Args:
            dir (str): The directory containing the PDFs to embed.
            aleph_alpha_token (str): The Aleph Alpha API token.

        Returns:
            None
        """
        vector_db: Qdrant = get_db_connection(collection_name=self.collection_name, aleph_alpha_token=self.aleph_alpha_token)

        if file_ending == "*.pdf":
            loader = DirectoryLoader(directory, glob=file_ending, loader_cls=PyPDFium2Loader)
        elif file_ending == "*.txt":
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

        vector_db.add_texts(texts=text_list, metadatas=metadata_list)

        vector_db.add_texts(texts=text_list, metadatas=metadata_list)

        logger.info("SUCCESS: Texts embedded.")

    def search(self, search: SearchRequest) -> List[Tuple[LangchainDocument, float]]:
        """Searches the Aleph Alpha service for similar documents.

        Args:
            aleph_alpha_token (str): Aleph Alpha API Token.
            query (str): The query that should be searched for.
            amount (int, optional): The number of documents to return. Defaults to 1.

        Returns
            List[Tuple[Document, float]]: A list of tuples containing the documents and their similarity scores.
        """
        if not query:
            raise ValueError("Query cannot be None or empty.")
        if amount < 1:
            raise ValueError("Amount must be greater than 0.")

        # TODO: FILTER
        try:

            vector_db: Qdrant = self.get_db_connection()
            docs = vector_db.similarity_search_with_score(query=searh.query, k=search.amount, score_threshold=search.filtering.threshold)
            logger.info("SUCCESS: Documents found.")

        except Exception as e:

            logger.error(f"ERROR: Failed to search documents: {e}")
            raise Exception(f"Failed to search documents: {e}") from e

        return docs

    def rag(self, rag_request: RAGRequest) -> tuple:
        """QA takes a list of documents and returns a list of answers.

        Args:
            aleph_alpha_token (str): The Aleph Alpha API token.
            documents (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
            query (str): The query to ask.
            summarization (bool, optional): Whether to use summarization. Defaults to False.

        Returns:
            Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]: A tuple containing the answer, the prompt, and the metadata for the documents.
        """
        # if the list of documents contains only one document extract the text directly
        if len(documents) == 1:
            text = documents[0][0].page_content
            meta_data = documents[0][0].metadata

        else:
            # extract the text from the documents
            texts = [doc[0].page_content for doc in documents]
            if summarization:
                text = "".join(self.summarize_text(t) for t in texts)
            else:
                # combine the texts to one text
                text = " ".join(texts)
            meta_data = [doc[0].metadata for doc in documents]

        # load the prompt
        prompt = generate_prompt("aleph_alpha_qa.j2", text=text, query=query)

        try:
            # call the luminous api
            answer = self.send_completion_request(prompt, aleph_alpha_token)

        except ValueError as e:
            # if the code is PROMPT_TOO_LONG, split it into chunks
            if e.args[0] == "PROMPT_TOO_LONG":
                logger.info("Prompt too long. Summarizing.")

                # summarize the text
                short_text = self.summarize_text(text)

                # generate the prompt
                prompt = generate_prompt("aleph_alpha_qa.j2", text=short_text, query=query)

                # call the luminous api
                answer = self.send_completion_request(prompt)

        # extract the answer
        return answer, prompt, meta_data

    @load_config(location="config/ai/aleph_alpha.yml")
    def explain_qa(self, document: LangchainDocument, explain_threshold: float, query: str, cfg: DictConfig):
        """Explian QA WIP."""
        text = document[0][0].page_content
        meta_data = document[0][0].metadata

        # load the prompt
        prompt = generate_prompt("aleph_alpha_qa.j2", text=text, query=query)

        answer = self.send_completion_request(prompt, aleph_alpha_token)

        exp_req = ExplanationRequest(Prompt.from_text(prompt), answer, control_factor=0.1, prompt_granularity="sentence", normalize=True)
        client = Client(token=aleph_alpha_token)

        response_explain = client.explain(exp_req, model=cfg.aleph_alpha_completion.model)
        explanations = response_explain.explanations[0].items[0].scores

        # if all of the scores are belo 0.7 raise an error
        if all(item.score < explain_threshold for item in explanations):
            raise ValueError("All scores are below explain_threshold.")

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

    def process_documents_aleph_alpha(self, folder: str, type: str) -> List[str]:
        """Process the documents in the given folder.

        Args:
            folder (str): Folder where the documents are located.
            token (str): The Aleph Alpha API Token.
            type (str): The type of the documents.

        Raises:
            ValueError: If the type is not one of 'qa', 'summarization', or 'invoice'.
        """
        # load the documents
        loader = DirectoryLoader(folder, glob="*.pdf", loader_cls=PyPDFium2Loader)

        # load the documents
        docs = loader.load()

        # load the correct prompt
        match type:
            case "qa":
                raise NotImplementedError
            case "summarization":
                raise NotImplementedError
            case "invoice":
                # load the prompt
                prompt_name = "aleph-alpha-invoice.j2"
            case _:
                raise ValueError("Type must be one of 'qa', 'summarization', or 'invoice'.")

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

    def custom_completion_prompt_aleph_alpha(
        self,
        prompt: str,
        model: str = "luminous-extended-control",
        max_tokens: int = 256,
        stop_sequences: List[str] = ["###"],
        temperature: float = 0,
    ) -> str:
        """This method sents a custom completion request to the Aleph Alpha API.

        Args:
            token (str): The token for the Aleph Alpha API.
            prompt (str): The prompt to be sent to the API.

        Raises:
            ValueError: Error if their are no completions or the completion is empty or the prompt and tokenis empty.
        """
        if not prompt:
            raise ValueError("Prompt cannot be None or empty.")
        if not token:
            raise ValueError("Token cannot be None or empty.")

        client = Client(token=token)

        request = CompletionRequest(prompt=Prompt.from_text(prompt), maximum_tokens=max_tokens, stop_sequences=stop_sequences, temperature=temperature)
        response = client.complete(request, model=model)

        # ensure that the response is not empty
        if not response.completions:
            raise ValueError("Response is empty.")

        # ensure that the completion is not empty
        if not response.completions[0].completion:
            raise ValueError("Completion is empty.")

        return str(response.completions[0].completion)


if __name__ == "__main__":
    token = os.getenv("ALEPH_ALPHA_API_KEY")

    if not token:
        raise ValueError("Token cannot be None or empty.")

    aa_service = AlephAlphaService()

    aa_service.embed_documents("tests/resources/")
    # open the text file and read the text
    DOCS = aa_service.search(search=SearchRequest(query="What are Attentions?"))
    logger.info(DOCS)

    answer, prompt, meta_data = aa_service.rag(rag_request=RAGRequest(documents=DOCS, query="What are Attentions?"))
