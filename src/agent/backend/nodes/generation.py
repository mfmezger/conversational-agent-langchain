"""Generation nodes for the graph."""

from collections.abc import Sequence

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, convert_to_messages
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from agent.backend.prompts import COHERE_RESPONSE_TEMPLATE, RESPONSE_TEMPLATE
from agent.backend.state import AgentState
from agent.utils.utility import format_docs_for_citations


def get_chat_history(messages: Sequence[BaseMessage]) -> list:
    """Append the chat history to the messages."""
    return [
        {"content": message.content, "role": message.type}
        for message in messages
        if (isinstance(message, AIMessage) and not message.tool_calls) or isinstance(message, HumanMessage)
    ]


def generate_response(state: AgentState, model: LanguageModelLike, prompt_template: str) -> AgentState:
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
            "chat_history": get_chat_history(convert_to_messages(state["messages"][:-1])),
        }
    )
    return {
        "messages": [synthesized_response],
    }


def generate_response_default(state: AgentState, *, llm: LanguageModelLike) -> AgentState:
    """Generate a response using non cohere model."""
    return generate_response(state, llm, RESPONSE_TEMPLATE)


def generate_response_cohere(state: AgentState, *, cohere_llm: LanguageModelLike | None, llm: LanguageModelLike) -> AgentState:
    """Generate a response using Cohere's grounded generation with native document support."""
    if not cohere_llm:
        logger.warning("Cohere API key not configured, falling back to default response synthesizer")
        return generate_response_default(state, llm=llm)

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
            "chat_history": get_chat_history(convert_to_messages(state["messages"][:-1])),
        },
        documents=cohere_documents,
    )
    return {
        "messages": [synthesized_response],
    }
