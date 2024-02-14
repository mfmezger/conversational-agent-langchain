"""GPT4ALL Backend Service."""
from typing import List, Tuple

from dotenv import load_dotenv
from gpt4all import GPT4All
from langchain.docstore.document import Document
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBase
from agent.utils.utility import generate_prompt
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
        prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text, language="de")

        model = GPT4All(self.cfg.gpt4all_completion.completion_model)

        return model.generate(prompt, max_tokens=300)

    def completion_text_gpt4all(self, prompt: str) -> str:
        """Complete text with GPT4ALL.

        Args:
            text (str): The text as basic input.
            query (str): The query to be inserted into the template.

        Returns:
            str: The completed text.
        """
        model = GPT4All(self.cfg.gpt4all_completion.completion_model)

        return model.generate(prompt, max_tokens=100)

    def custom_completion_prompt_gpt4all(self, prompt: str, model: str = "orca-mini-3b.ggmlv3.q4_0.bin", max_tokens: int = 256, temperature: float = 0) -> str:
        """This method sents a custom completion request to the Aleph Alpha API.

        Args:
            token (str): The token for the Aleph Alpha API.
            prompt (str): The prompt to be sent to the API.

        Raises:
            ValueError: Error if their are no completions or the completion is empty or the prompt and tokenis empty.
        """
        if not prompt:
            raise ValueError("Prompt cannot be None or empty.")

        output = (GPT4All(model)).generate(prompt, max_tokens=max_tokens, temp=temperature)

        return str(output)

    def search(self, query: str, amount: int, threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """Searches the documents in the Qdrant DB with a specific query.

        Args:
            open_ai_token (str): The OpenAI API token.
            query (str): The question for which documents should be searched.

        Returns:
            List[Tuple[Document, float]]: A list of search results, where each result is a tuple
            containing a Document object and a float score.
        """
        docs = self.vector_db.similarity_search_with_score(query=query, k=amount, score_threshold=threshold)
        logger.info("SUCCESS: Documents found.")
        return docs

    def rag(self, documents: list[tuple[Document, float]], query: str, summarization: bool = False, language: str = "de"):
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
                text = "".join(summarize_text_gpt4all(t) for t in texts)
            else:
                # combine the texts to one text
                text = " ".join(texts)
            meta_data = [doc[0].metadata for doc in documents]

        # load the prompt
        prompt = generate_prompt("gpt4all-completion.j2", text=text, query=query, language=language)

        try:

            # call the luminous api
            logger.info("starting completion")
            answer = self.completion_text_gpt4all(prompt)
            logger.info(f"completion done with answer {answer}")

        except ValueError as e:
            # if the code is PROMPT_TOO_LONG, split it into chunks
            if e.args[0] == "PROMPT_TOO_LONG":
                logger.info("Prompt too long. Summarizing.")

                # summarize the text
                short_text = self.summarize_text_gpt4all(text)

                # generate the prompt
                prompt = generate_prompt("gpt4all-completion.j2", text=short_text, query=query, language=language)

                # call the luminous api
                answer = self.completion_text_gpt4all(prompt)

        # extract the answer
        return answer, prompt, meta_data


if __name__ == "__main__":

    gpt4all_service = GPT4AllService(collection_name="gpt4all", token="")

    gpt4all_service.embed_documents(directory="data")

    print(f'Summary: {gpt4all_service.summarize_text(text="Was ist Attention?")}')

    print(f'Completion: {gpt4all_service.completion_text_gpt4all(prompt="Was ist Attention?")}')

    docs = gpt4all_service.search(query="Was ist Attention?", amount=1)

    logger.info(f"Documents: {docs}")

    answer, prompt, meta_data = gpt4all_service.rag(documents=docs, query="Was ist das?")

    logger.info(f"Answer: {answer}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Metadata: {meta_data}")
