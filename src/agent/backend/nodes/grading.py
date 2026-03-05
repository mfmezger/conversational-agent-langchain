"""Grading node for the graph."""

from typing import Literal

from langchain_core.language_models import LanguageModelLike
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig

from agent.backend.prompts import GRADER_TEMPLATE
from agent.backend.state import AgentState, Grade


def grade_documents(
    state: AgentState,
    config: RunnableConfig,
    *,
    llm: LanguageModelLike,
) -> Literal["response_synthesizer", "response_synthesizer_cohere", "rewrite_query"]:
    """Grade the retrieved documents holistically."""
    model = llm.with_config(tags=["nostream"])
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
        return route_to_response_synthesizer(state, config)

    return "rewrite_query"


def route_to_response_synthesizer(state: AgentState, config: RunnableConfig) -> Literal["response_synthesizer", "response_synthesizer_cohere"]:  # noqa: ARG001
    """Route to the appropriate response synthesizer based on the config."""
    model_name = config.get("configurable", {}).get("model_name", "gemini")
    if model_name == "cohere_command":
        return "response_synthesizer_cohere"
    else:
        return "response_synthesizer"
