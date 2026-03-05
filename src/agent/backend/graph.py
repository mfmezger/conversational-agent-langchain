"""Defining the graph."""

from collections.abc import Sequence
from functools import partial
from typing import Literal

from langchain_cohere import ChatCohere
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, StateGraph
from loguru import logger

from agent.backend.nodes.generation import generate_response_cohere, generate_response_default
from agent.backend.nodes.grading import grade_documents
from agent.backend.nodes.retrieval import retrieve_documents, retrieve_documents_with_chat_history
from agent.backend.nodes.rewrite import rewrite_query
from agent.backend.state import AgentState
from agent.utils.config import Config
from agent.utils.reranker import get_reranker

settings = Config()


def get_chat_history(messages: Sequence[BaseMessage]) -> list:
    """Append the chat history to the messages."""
    return [
        {"content": message.content, "role": message.type}
        for message in messages
        if (isinstance(message, AIMessage) and not message.tool_calls) or isinstance(message, HumanMessage)
    ]

def route_to_retriever(
    state: AgentState,
) -> Literal["retriever", "retriever_with_chat_history"]:
    """Route to the appropriate retriever based on the state."""
    if len(state["messages"]) == 1:
        return "retriever"
    return "retriever_with_chat_history"

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

    def rerank_documents(self, state: AgentState) -> AgentState:
        """Rerank retrieved documents to improve relevance."""
        if not state.get("documents"):
            return state

        reranked_docs = self.reranker(state["documents"], state["query"])
        logger.info(f"Reranked documents: {len(state['documents'])} -> {len(reranked_docs)}")
        return {"documents": reranked_docs, "query": state.get("query"), "retry_count": state.get("retry_count", 0), "messages": []}

    def grade_documents_node(self, state: AgentState, config: RunnableConfig) -> Literal["response_synthesizer", "response_synthesizer_cohere", "rewrite_query"]:
        """Grade the retrieved documents holistically."""
        return grade_documents(state, config, self.llm)

    def build_graph(self) -> StateGraph:
        """Build the graph for the agent."""
        workflow = StateGraph(state_schema=AgentState)

        # define nodes
        workflow.add_node("retriever", retrieve_documents)
        workflow.add_node("retriever_with_chat_history", partial(retrieve_documents_with_chat_history, llm=self.llm, get_chat_history_func=get_chat_history))
        workflow.add_node("reranker", self.rerank_documents)
        workflow.add_node("rewrite_query", partial(rewrite_query, llm=self.llm))
        workflow.add_node("response_synthesizer", partial(generate_response_default, llm=self.llm, get_chat_history_func=get_chat_history))
        workflow.add_node("response_synthesizer_cohere", partial(generate_response_cohere, cohere_llm=self.cohere_llm, llm=self.llm, get_chat_history_func=get_chat_history))

        # set entry point to retrievers
        workflow.set_conditional_entry_point(path=route_to_retriever)

        # connect retrievers to reranker
        workflow.add_edge("retriever", "reranker")
        workflow.add_edge("retriever_with_chat_history", "reranker")

        # connect reranker to grader
        workflow.add_conditional_edges(
            source="reranker",
            path=self.grade_documents_node,
            path_map={"response_synthesizer": "response_synthesizer", "response_synthesizer_cohere": "response_synthesizer_cohere", "rewrite_query": "rewrite_query"},
        )

        # connect rewriter back to retriever (loop)
        # Note: We always route back to basic retriever because we have a standalone query now
        workflow.add_edge("rewrite_query", "retriever")

        # connect synthesizers to terminal node
        workflow.add_edge(start_key="response_synthesizer", end_key=END)
        workflow.add_edge(start_key="response_synthesizer_cohere", end_key=END)

        return workflow.compile()
