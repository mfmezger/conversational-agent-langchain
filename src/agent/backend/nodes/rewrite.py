"""Rewrite query node."""

from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from loguru import logger

from agent.backend.prompts import REWRITE_TEMPLATE
from agent.backend.state import AgentState


def rewrite_query(state: AgentState, llm: LanguageModelLike) -> AgentState:
    """Rewrite the query to improve retrieval."""
    model = llm.with_config(tags=["nostream"])
    prompt = PromptTemplate(
        template=REWRITE_TEMPLATE,
        input_variables=["question"],
    )
    chain = prompt | model | StrOutputParser()
    new_query = chain.invoke({"question": state["query"]})
    logger.info(f"Rewritten query: {new_query}")
    return {"query": new_query, "retry_count": state.get("retry_count", 0) + 1, "messages": [], "documents": state.get("documents", [])}
