"""Generation node."""

from collections.abc import Callable

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import convert_to_messages
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from agent.backend.prompts import COHERE_RESPONSE_TEMPLATE, RESPONSE_TEMPLATE
from agent.backend.state import AgentState
from agent.utils.utility import format_docs_for_citations


def generate_response(state: AgentState, model: LanguageModelLike, prompt_template: str, get_chat_history_func: Callable) -> AgentState:
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
            "chat_history": get_chat_history_func(convert_to_messages(state["messages"][:-1])),
        }
    )
    return {
        "messages": [synthesized_response],
        "query": state.get("query"),
        "documents": state.get("documents", []),
        "retry_count": state.get("retry_count", 0),
    }


def generate_response_default(state: AgentState, llm: LanguageModelLike, get_chat_history_func: Callable) -> AgentState:
    """Generate a response using non cohere model."""
    return generate_response(state, llm, RESPONSE_TEMPLATE, get_chat_history_func)


def generate_response_cohere(state: AgentState, cohere_llm: LanguageModelLike | None, llm: LanguageModelLike, get_chat_history_func: Callable) -> AgentState:
    """Generate a response using Cohere's grounded generation with native document support."""
    if not cohere_llm:
        logger.warning("Cohere API key not configured, falling back to default response synthesizer")
        return generate_response_default(state, llm, get_chat_history_func)

    # Convert LangChain Documents to Cohere's expected format
    cohere_documents = [{"text": doc.page_content, "title": doc.metadata.get("source", f"Document {i + 1}")} for i, doc in enumerate(state["documents"])]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COHERE_RESPONSE_TEMPLATE),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )
    response_synthesizer = prompt | cohere_llm

    # Pass documents via invoke - Cohere handles grounding natively
    synthesized_response = response_synthesizer.invoke(
        {
            "question": state["query"],
            "chat_history": get_chat_history_func(convert_to_messages(state["messages"][:-1])),
        },
        documents=cohere_documents,
    )
    return {
        "messages": [synthesized_response],
            "query": state.get("query"),
            "documents": state.get("documents", []),
            "retry_count": state.get("retry_count", 0),
    }
