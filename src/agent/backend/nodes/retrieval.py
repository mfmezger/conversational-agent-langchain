"""Retrieval nodes for the graph."""

from collections.abc import Callable, Sequence

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, convert_to_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from loguru import logger

from agent.backend.prompts import REPHRASE_TEMPLATE
from agent.backend.state import AgentState
from agent.utils.config import Config
from agent.utils.retriever import get_retriever


def get_chat_history(messages: Sequence[BaseMessage]) -> list:
    """Append the chat history to the messages."""
    return [
        {"content": message.content, "role": message.type}
        for message in messages
        if (isinstance(message, AIMessage) and not message.tool_calls) or isinstance(message, HumanMessage)
    ]


def retrieve_documents(state: AgentState, config: RunnableConfig, *, cfg: Config) -> AgentState:
    """Retrieve documents from the retriever."""
    # Dynamic k: Increase k if retrying
    retry_count = state.get("retry_count", 0)
    k = cfg.retrieval_k if retry_count == 0 else cfg.retrieval_k_retry

    retriever = get_retriever(k=k, collection_name=config["metadata"]["collection_name"])
    messages = convert_to_messages(messages=state["messages"])
    # If query was rewritten, use state["query"], otherwise use last message
    query = state.get("query") or messages[-1].content

    relevant_documents = retriever.invoke(query)
    if not relevant_documents:
        logger.info(f"No relevant documents found for the query: {query}")

    return {"query": query, "documents": relevant_documents, "retry_count": retry_count}


def retrieve_documents_with_chat_history(state: AgentState, config: RunnableConfig, *, cfg: Config, llm: LanguageModelLike) -> AgentState:
    """Retrieve documents from the retriever with chat history."""
    # Dynamic k: Increase k if retrying
    retry_count = state.get("retry_count", 0)
    k = cfg.retrieval_k if retry_count == 0 else cfg.retrieval_k_retry

    retriever = get_retriever(k=k, collection_name=config["metadata"]["collection_name"])
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
        relevant_documents = retriever_with_condensed_question.invoke({"question": query, "chat_history": get_chat_history(messages[:-1])})
        return {"query": query, "documents": relevant_documents, "retry_count": retry_count}
    else:
        # If we are looping, we already have a rewritten query in state["query"]
        # So we just use basic retrieval on that
        return retrieve_documents(state, config, cfg=cfg)


def rerank_documents(state: AgentState, *, reranker: Callable) -> AgentState:
    """Rerank retrieved documents to improve relevance."""
    if not state["documents"]:
        return state

    reranked_docs = reranker(state["documents"], state["query"])
    logger.info(f"Reranked documents: {len(state['documents'])} -> {len(reranked_docs)}")
    return {"documents": reranked_docs}
