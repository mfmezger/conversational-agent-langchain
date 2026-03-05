"""Retrieval node."""

from collections.abc import Callable
from typing import cast

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import convert_to_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from loguru import logger

from agent.backend.prompts import REPHRASE_TEMPLATE
from agent.backend.state import AgentState
from agent.utils.config import Config
from agent.utils.retriever import get_retriever

cfg = Config()


def retrieve_documents(state: AgentState, config: RunnableConfig) -> AgentState:
    """Retrieve documents from the retriever."""
    # Dynamic k: Increase k if retrying
    retry_count = state.get("retry_count", 0)
    k = cfg.retrieval_k if retry_count == 0 else cfg.retrieval_k_retry

    metadata = config.get("metadata", {})
    collection_name = metadata.get("collection_name", cfg.qdrant_collection_name)

    retriever = get_retriever(k=k, collection_name=collection_name)
    messages = convert_to_messages(messages=state["messages"])
    # If query was rewritten, use state["query"], otherwise use last message
    query = state.get("query") or messages[-1].content
    query = cast("str", query)

    relevant_documents = retriever.invoke(query)
    if not relevant_documents:
        logger.info(f"No relevant documents found for the query: {query}")

    return {"query": query, "documents": relevant_documents, "retry_count": retry_count, "messages": []}


def retrieve_documents_with_chat_history(state: AgentState, config: RunnableConfig, llm: LanguageModelLike, get_chat_history_func: Callable) -> AgentState:
    """Retrieve documents from the retriever with chat history."""
    # Dynamic k: Increase k if retrying
    retry_count = state.get("retry_count", 0)
    k = cfg.retrieval_k if retry_count == 0 else cfg.retrieval_k_retry

    metadata = config.get("metadata", {})
    collection_name = metadata.get("collection_name", cfg.qdrant_collection_name)

    retriever = get_retriever(k=k, collection_name=collection_name)
    model = llm.with_config(tags=["nostream"])

    condense_queston_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (condense_queston_prompt | model | StrOutputParser()).with_config(
        run_name="CondenseQuestion",
    )

    messages = convert_to_messages(messages=state["messages"])
    # If query was rewritten, use state["query"], otherwise use last message
    if not state.get("query"):
        query = messages[-1].content
        retriever_with_condensed_question = condense_question_chain | retriever
        relevant_documents = retriever_with_condensed_question.invoke({"question": query, "chat_history": get_chat_history_func(messages[:-1])})
        return {"query": query, "documents": relevant_documents, "retry_count": retry_count, "messages": []}

    # If we are looping, we already have a rewritten query in state["query"]
    # So we just use basic retrieval on that
    return retrieve_documents(state, config)
