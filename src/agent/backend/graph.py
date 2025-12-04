"""Defining the graph."""

from collections.abc import Sequence
from typing import Annotated, Literal, TypedDict

from langchain_cohere import ChatCohere
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, convert_to_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, StateGraph, add_messages
from loguru import logger
from pydantic import BaseModel, Field

from agent.backend.prompts import COHERE_RESPONSE_TEMPLATE, GRADER_TEMPLATE, REPHRASE_TEMPLATE, RESPONSE_TEMPLATE, REWRITE_TEMPLATE
from agent.utils.config import Config
from agent.utils.reranker import get_reranker
from agent.utils.retriever import get_retriever
from agent.utils.utility import format_docs_for_citations

GEMINI_MODEL_KEY = "gemini"
COHERE_MODEL_KEY = "cohere_command"


settings = Config()


class AgentState(TypedDict):
    """State of the Agent."""

    query: str
    documents: list[Document]
    messages: Annotated[list[BaseMessage], add_messages]
    retry_count: int


class Grade(BaseModel):
    """Binary score for relevance check."""

    is_relevant: bool = Field(description="True if the documents are relevant to the question, False otherwise")


class Graph:
    """The LangGraph Graph."""

    def __init__(self) -> None:
        """Initialize the Graph."""
        self.cfg = settings

        # define models
        self.llm = ChatLiteLLM(model_name=self.cfg.model_name, streaming=True)

        # Initialize Cohere model for grounded generation (if API key is available)
        self.cohere_llm: ChatCohere | None = None
        if self.cfg.cohere_api_key:
            self.cohere_llm = ChatCohere(
                cohere_api_key=self.cfg.cohere_api_key,
                model=self.cfg.cohere_model_name,
                streaming=True,
            )

        # Initialize reranker
        self.reranker = get_reranker(
            provider=self.cfg.rerank_provider,
            top_k=self.cfg.rerank_top_k,
            cohere_api_key=self.cfg.cohere_api_key,
        )

    def retrieve_documents(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Retrieve documents from the retriever."""
        # Dynamic k: Increase k if retrying
        retry_count = state.get("retry_count", 0)
        k = self.cfg.retrieval_k if retry_count == 0 else self.cfg.retrieval_k_retry

        retriever = get_retriever(k=k, collection_name=config["metadata"]["collection_name"])
        messages = convert_to_messages(messages=state["messages"])
        # If query was rewritten, use state["query"], otherwise use last message
        query = state.get("query") or messages[-1].content

        relevant_documents = retriever.invoke(query)
        if not relevant_documents:
            logger.info(f"No relevant documents found for the query: {query}")

        return {"query": query, "documents": relevant_documents, "retry_count": retry_count}

    def retrieve_documents_with_chat_history(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Retrieve documents from the retriever with chat history."""
        # Dynamic k: Increase k if retrying
        retry_count = state.get("retry_count", 0)
        k = self.cfg.retrieval_k if retry_count == 0 else self.cfg.retrieval_k_retry

        retriever = get_retriever(k=k, collection_name=config["metadata"]["collection_name"])
        model = self.llm.with_config(tags=["nostream"])

        condense_queston_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
        condense_question_chain = (condense_queston_prompt | model | StrOutputParser()).with_config(
            run_name="CondenseQuestion",
        )

        messages = convert_to_messages(messages=state["messages"])
        # If query was rewritten, use state["query"], otherwise use last message
        if not state.get("query"):
            query = messages[-1].content
            retriever_with_condensed_question = condense_question_chain | retriever
            relevant_documents = retriever_with_condensed_question.invoke({"question": query, "chat_history": self.get_chat_history(messages[:-1])})
            return {"query": query, "documents": relevant_documents, "retry_count": retry_count}
        else:
            # If we are looping, we already have a rewritten query in state["query"]
            # So we just use basic retrieval on that
            return self.retrieve_documents(state, config)

    def rerank_documents(self, state: AgentState) -> AgentState:
        """Rerank retrieved documents to improve relevance."""
        if not state["documents"]:
            return state

        reranked_docs = self.reranker(state["documents"], state["query"])
        logger.info(f"Reranked documents: {len(state['documents'])} -> {len(reranked_docs)}")
        return {"documents": reranked_docs}

    def grade_documents(self, state: AgentState, config: RunnableConfig) -> Literal["response_synthesizer", "response_synthesizer_cohere", "rewrite_query"]:
        """Grade the retrieved documents holistically."""
        model = self.llm.with_config(tags=["nostream"])
        structured_model = model.with_structured_output(Grade)

        prompt = PromptTemplate(
            template=GRADER_TEMPLATE,
            input_variables=["documents", "question"],
        )
        chain = prompt | structured_model

        # Format documents for the grader
        docs_text = "\n\n".join([f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(state["documents"])])

        grade: Grade = chain.invoke({"documents": docs_text, "question": state["query"]})

        # If graded as relevant, or we hit max retries, generate
        if grade.is_relevant or state.get("retry_count", 0) >= 2:
            return self.route_to_response_synthesizer(state, config)

        return "rewrite_query"

    def rewrite_query(self, state: AgentState) -> AgentState:
        """Rewrite the query to improve retrieval."""
        model = self.llm.with_config(tags=["nostream"])
        prompt = PromptTemplate(
            template=REWRITE_TEMPLATE,
            input_variables=["question"],
        )
        chain = prompt | model | StrOutputParser()
        new_query = chain.invoke({"question": state["query"]})
        logger.info(f"Rewritten query: {new_query}")
        return {"query": new_query, "retry_count": state.get("retry_count", 0) + 1}

    def route_to_retriever(
        self,
        state: AgentState,
    ) -> Literal["retriever", "retriever_with_chat_history"]:
        """Route to the appropriate retriever based on the state."""
        if len(state["messages"]) == 1:
            return "retriever"
        else:
            return "retriever_with_chat_history"

    def get_chat_history(self, messages: Sequence[BaseMessage]) -> list:
        """Append the chat history to the messages."""
        return [
            {"content": message.content, "role": message.type}
            for message in messages
            if (isinstance(message, AIMessage) and not message.tool_calls) or isinstance(message, HumanMessage)
        ]

    def generate_response(self, state: AgentState, model: LanguageModelLike, prompt_template: str) -> AgentState:
        """Create a response from the model."""
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
                "question": state["query"],
                "context": format_docs_for_citations(state["documents"]),
                "chat_history": self.get_chat_history(convert_to_messages(state["messages"][:-1])),
            }
        )
        return {
            "messages": [synthesized_response],
        }

    def generate_response_default(self, state: AgentState) -> AgentState:
        """Generate a response using non cohere model."""
        return self.generate_response(state, self.llm, RESPONSE_TEMPLATE)

    def generate_response_cohere(self, state: AgentState) -> AgentState:
        """Generate a response using Cohere's grounded generation with native document support."""
        if not self.cohere_llm:
            logger.warning("Cohere API key not configured, falling back to default response synthesizer")
            return self.generate_response_default(state)

        # Convert LangChain Documents to Cohere's expected format
        cohere_documents = [{"text": doc.page_content, "title": doc.metadata.get("source", f"Document {i + 1}")} for i, doc in enumerate(state["documents"])]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", COHERE_RESPONSE_TEMPLATE),
                ("placeholder", "{chat_history}"),
                ("human", "{question}"),
            ]
        )
        response_synthesizer = prompt | self.cohere_llm

        # Pass documents via invoke - Cohere handles grounding natively
        synthesized_response = response_synthesizer.invoke(
            {
                "question": state["query"],
                "chat_history": self.get_chat_history(convert_to_messages(state["messages"][:-1])),
            },
            documents=cohere_documents,
        )
        return {
            "messages": [synthesized_response],
        }

    def route_to_response_synthesizer(self, state: AgentState, config: RunnableConfig) -> Literal["response_synthesizer", "response_synthesizer_cohere"]:  # noqa: ARG002
        """Route to the appropriate response synthesizer based on the config."""
        model_name = config.get("configurable", {}).get("model_name", GEMINI_MODEL_KEY)
        if model_name == COHERE_MODEL_KEY:
            return "response_synthesizer_cohere"
        else:
            return "response_synthesizer"

    def build_graph(self) -> StateGraph:
        """Build the graph for the agent."""
        workflow = StateGraph(state_schema=AgentState)

        # define nodes
        workflow.add_node("retriever", self.retrieve_documents)
        workflow.add_node("retriever_with_chat_history", self.retrieve_documents_with_chat_history)
        workflow.add_node("reranker", self.rerank_documents)
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("response_synthesizer", self.generate_response_default)
        workflow.add_node("response_synthesizer_cohere", self.generate_response_cohere)

        # set entry point to retrievers
        workflow.set_conditional_entry_point(path=self.route_to_retriever)

        # connect retrievers to reranker
        workflow.add_edge("retriever", "reranker")
        workflow.add_edge("retriever_with_chat_history", "reranker")

        # connect reranker to grader
        workflow.add_conditional_edges(
            source="reranker",
            path=self.grade_documents,
            path_map={"response_synthesizer": "response_synthesizer", "response_synthesizer_cohere": "response_synthesizer_cohere", "rewrite_query": "rewrite_query"},
        )

        # connect rewriter back to retriever (loop)
        # Note: We always route back to basic retriever because we have a standalone query now
        workflow.add_edge("rewrite_query", "retriever")

        # connect synthesizers to terminal node
        workflow.add_edge(start_key="response_synthesizer", end_key=END)
        workflow.add_edge(start_key="response_synthesizer_cohere", end_key=END)

        return workflow.compile()
