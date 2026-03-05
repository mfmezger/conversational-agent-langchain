"""Grading node."""

from typing import Literal

from langchain_core.language_models import LanguageModelLike
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig

from agent.backend.prompts import GRADER_TEMPLATE
from agent.backend.state import AgentState, Grade

# Need to redefine these here to avoid circular imports or we can just use the strings
GEMINI_MODEL_KEY = "gemini"
COHERE_MODEL_KEY = "cohere_command"

def route_to_response_synthesizer(config: RunnableConfig) -> Literal["response_synthesizer", "response_synthesizer_cohere"]:
    """Route to the appropriate response synthesizer based on the config."""
    model_name = config.get("configurable", {}).get("model_name", GEMINI_MODEL_KEY)
    if model_name == COHERE_MODEL_KEY:
        return "response_synthesizer_cohere"
    return "response_synthesizer"

def grade_documents(state: AgentState, config: RunnableConfig, llm: LanguageModelLike) -> Literal["response_synthesizer", "response_synthesizer_cohere", "rewrite_query"]:
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
        return route_to_response_synthesizer(config)

    return "rewrite_query"
