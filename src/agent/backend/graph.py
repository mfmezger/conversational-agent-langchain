"""Defining the graph."""

import functools
from typing import Literal

from langchain_cohere import ChatCohere
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, StateGraph

from agent.backend.nodes.generation import generate_response_cohere, generate_response_default
from agent.backend.nodes.grading import grade_documents
from agent.backend.nodes.retrieval import rerank_documents, retrieve_documents, retrieve_documents_with_chat_history
from agent.backend.nodes.rewrite import rewrite_query
from agent.backend.state import AgentState
from agent.utils.config import Config
from agent.utils.reranker import get_reranker

GEMINI_MODEL_KEY = "gemini"
COHERE_MODEL_KEY = "cohere_command"


settings = Config()


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

    def route_to_retriever(
        self,
        state: AgentState,
    ) -> Literal["retriever", "retriever_with_chat_history"]:
        """Route to the appropriate retriever based on the state."""
        if len(state["messages"]) == 1:
            return "retriever"
        else:
            return "retriever_with_chat_history"

    def build_graph(self) -> StateGraph:
        """Build the graph for the agent."""
        workflow = StateGraph(state_schema=AgentState)

        # define nodes
        workflow.add_node("retriever", functools.partial(retrieve_documents, cfg=self.cfg))
        workflow.add_node("retriever_with_chat_history", functools.partial(retrieve_documents_with_chat_history, cfg=self.cfg, llm=self.llm))
        workflow.add_node("reranker", functools.partial(rerank_documents, reranker=self.reranker))
        workflow.add_node("rewrite_query", functools.partial(rewrite_query, llm=self.llm))
        workflow.add_node("response_synthesizer", functools.partial(generate_response_default, llm=self.llm))
        workflow.add_node("response_synthesizer_cohere", functools.partial(generate_response_cohere, llm=self.llm, cohere_llm=self.cohere_llm))

        # set entry point to retrievers
        workflow.set_conditional_entry_point(path=self.route_to_retriever)

        # connect retrievers to reranker
        workflow.add_edge("retriever", "reranker")
        workflow.add_edge("retriever_with_chat_history", "reranker")

        # connect reranker to grader
        workflow.add_conditional_edges(
            source="reranker",
            path=functools.partial(grade_documents, llm=self.llm),
            path_map={"response_synthesizer": "response_synthesizer", "response_synthesizer_cohere": "response_synthesizer_cohere", "rewrite_query": "rewrite_query"},
        )

        # connect rewriter back to retriever (loop)
        # Note: We always route back to basic retriever because we have a standalone query now
        workflow.add_edge("rewrite_query", "retriever")

        # connect synthesizers to terminal node
        workflow.add_edge(start_key="response_synthesizer", end_key=END)
        workflow.add_edge(start_key="response_synthesizer_cohere", end_key=END)

        return workflow.compile()
