"""Defining the graph."""

from collections.abc import Sequence
from typing import Annotated, Literal, TypedDict

from langchain_cohere import CohereEmbeddings
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, convert_to_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig, chain
from langchain_qdrant import Qdrant
from langgraph.graph import END, StateGraph, add_messages
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from ultra_simple_config import load_config

from agent.backend.prompts import COHERE_RESPONSE_TEMPLATE, REPHRASE_TEMPLATE, RESPONSE_TEMPLATE
from agent.utils.config import Settings
from agent.utils.utility import format_docs_for_citations

OPENAI_MODEL_KEY = "openai_gpt_3_5_turbo"
COHERE_MODEL_KEY = "cohere_command"
OLLAMA_MODEL_KEY = "ollama_llama8b3.1"

settings = Settings()


class AgentState(TypedDict):
    """State of the Agent."""

    query: str
    documents: list[Document]
    messages: Annotated[list[BaseMessage], add_messages]


@load_config("config/litellm.yml")
def load_litellm_config(cfg: DictConfig) -> DictConfig:
    """Wrapper to load the config.

    Args:
    ----
        cfg (DictConfig): The Config as dict.

    Returns:
    -------
        DictConfig: The configutration for Litellm.

    """
    return cfg


class Graph:
    """The LangGraph Graph."""

    def __init__(self) -> None:
        """Initialize the Graph."""
        # first load the config
        self.cfg = load_litellm_config()

        # define models
        self.llm = ChatLiteLLM(model=self.cfg.generation_llm.model_name)

    def get_score_retriever(self) -> BaseRetriever:
        """Get the Retriever.

        Returns
        -------
            BaseRetriever: _description_

        """
        embedding = CohereEmbeddings(model="embed-multilingual-v3.0")

        qdrant_client = QdrantClient(
            settings.qdrant_url,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            prefer_grpc=settings.qdrant_prefer_http,
        )

        vector_db = Qdrant(client=qdrant_client, collection_name="cohere", embeddings=embedding)

        @chain
        def retriever_with_score(query: str) -> list[Document]:
            """Defines a retriever that returns the score.

            Args:
            ----
                query (str): Query the user asks.

            Returns:
            -------
                list[Document]: List of Langchain Documents.

            """
            docs, scores = zip(*vector_db.similarity_search_with_score(query), strict=False)
            for doc, score in zip(docs, scores, strict=False):
                doc.metadata["score"] = score

            return docs

        return retriever_with_score

    def get_retriever() -> BaseRetriever:
        """Create a Vector Database retriever.

        Returns
        -------
            BaseRetriever: Qdrant + Cohere Embeddings Retriever

        """
        embedding = CohereEmbeddings(model="embed-multilingual-v3.0")

        qdrant_client = QdrantClient(
            settings.qdrant_url,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False,
        )

        vector_db = Qdrant(client=qdrant_client, collection_name="cohere", embeddings=embedding)
        return vector_db.as_retriever(search_kwargs={"k": 4})

    def retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve documents from the retriever.

        Args:
        ----
            state (AgentState): Graph State.

        Returns:
        -------
            AgentState: Modified Graph State.

        """
        retriever = self.get_retriever()
        messages = convert_to_messages(messages=state["messages"])
        query = messages[-1].content
        relevant_documents = retriever.invoke(query)
        return {"query": query, "documents": relevant_documents}

    def retrieve_documents_with_chat_history(self, state: AgentState) -> AgentState:
        """Retrieve documents from the retriever with chat history.

        Args:
        ----
            state (AgentState): Graph State.

        Returns:
        -------
            AgentState: Modified Graph State.

        """
        retriever = self.get_retriever()
        model = self.llm.with_config(tags=["nostream"])

        condense_queston_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
        condense_question_chain = (condense_queston_prompt | model | StrOutputParser()).with_config(
            run_name="CondenseQuestion",
        )

        messages = convert_to_messages(messages=state["messages"])
        query = messages[-1].content
        retriever_with_condensed_question = condense_question_chain | retriever
        relevant_documents = retriever_with_condensed_question.invoke({"question": query, "chat_history": self.get_chat_history(messages[:-1])})
        return {"query": query, "documents": relevant_documents}

    def route_to_retriever(
        self,
        state: AgentState,
    ) -> Literal["retriever", "retriever_with_chat_history"]:
        """Route to the appropriate retriever based on the state.

        Returns
        -------
            Literal["retriever", "retriever_with_chat_history"]: Choosen retriever method.

        """
        # at this point in the graph execution there is exactly one (i.e. first) message from the user,
        # so use basic retriever without chat history
        if len(state["messages"]) == 1:
            return "retriever"
        else:
            return "retriever_with_chat_history"

    def get_chat_history(self: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
        """Append the chat history to the messages.

        Args:
        ----
            messages (Sequence[BaseMessage]): Messages from the frontend.

        Returns:
        -------
            Sequence[BaseMessage]: Chat history as Langchain messages.

        """
        return [
            {"content": message.content, "role": message.type}
            for message in self
            if (isinstance(message, AIMessage) and not message.tool_calls) or isinstance(message, HumanMessage)
        ]

    def generate_response(self, state: AgentState, model: LanguageModelLike, prompt_template: str) -> AgentState:
        """Create a response from the model.

        Args:
        ----
            state (AgentState): Graph State.
            model (LanguageModelLike): Language Model.
            prompt_template (str): Template for the prompt.

        Returns:
        -------
            AgentState: Modified Graph State.

        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                ("placeholder", "{chat_history}"),
                ("human", "{question}"),
            ]
        )
        response_synthesizer = prompt | model
        synthesized_response = response_synthesizer.invoke(
            {
                "question": self["query"],
                "context": format_docs_for_citations(self["documents"]),
                # NOTE: we're ignoring the last message here, as it's going to contain the most recent
                # query and we don't want that to be included in the chat history
                "chat_history": self.get_chat_history(convert_to_messages(state["messages"][:-1])),
            }
        )
        return {
            "messages": [synthesized_response],
        }

    def generate_response_default(self, state: AgentState) -> AgentState:
        """Generate a response using non cohere model.

        Args:
        ----
            state (AgentState): Graph State.

        Returns:
        -------
            AgentState: Modified Graph State.

        """
        return self.generate_response(state, self.llm, RESPONSE_TEMPLATE)

    def generate_response_cohere(self, state: AgentState) -> AgentState:  # noqa: ARG002
        """Generate a response using the Cohere model.

        Args:
        ----
            state (AgentState): Graph State.

        Returns:
        -------
            AgentState: Modified Graph State.

        """
        model = self.llm.bind(documents=self["documents"])
        return self.generate_response(self, model, COHERE_RESPONSE_TEMPLATE)

    def route_to_response_synthesizer(self, state: AgentState, config: RunnableConfig) -> Literal["response_synthesizer", "response_synthesizer_cohere"]:  # noqa: ARG002
        """Route to the appropriate response synthesizer based on the config.

        Args:
        ----
            state (AgentState): Graph State.
            config (RunnableConfig): Runnable Config.


        Returns:
        -------
            Literal["response_synthesizer", "response_synthesizer_cohere"]: Choosen response synthesizer method.

        """
        model_name = config.get("configurable", {}).get("model_name", OPENAI_MODEL_KEY)
        if model_name == COHERE_MODEL_KEY:
            return "response_synthesizer_cohere"
        else:
            return "response_synthesizer"

    def build_graph(self) -> StateGraph:
        """Build the graph for the agent.

        Returns
        -------
            Graph: The generated graph for RAG.

        """
        workflow = StateGraph(state_schema=AgentState)

        # define nodes
        workflow.add_node("retriever", self.retrieve_documents)
        workflow.add_node("retriever_with_chat_history", self.retrieve_documents_with_chat_history)
        workflow.add_node("response_synthesizer", self.generate_response_default)
        workflow.add_node("response_synthesizer_cohere", self.generate_response_cohere)

        # set entry point to retrievers
        workflow.set_conditional_entry_point(path=self.route_to_retriever)

        # connect retrievers and response synthesizers
        workflow.add_conditional_edges(source="retriever", path=self.route_to_response_synthesizer)
        workflow.add_conditional_edges(source="retriever_with_chat_history", path=self.route_to_response_synthesizer)

        # connect synthesizers to terminal node
        workflow.add_edge(start_key="response_synthesizer", end_key=END)
        workflow.add_edge(start_key="response_synthesizer_cohere", end_key=END)

        return workflow.compile()

    # answer = graph.invoke({"messages": [{"role": "human", "content": "wer ist der vater von luke skywalker?"}, {"role": "assistant", "content": "Der Vater von Luke
    # Skywalker war Anakin Skywalker."}, {"role": "human", "content": "und wer ist seine mutter?"}]})
    # logger.info(answer)
