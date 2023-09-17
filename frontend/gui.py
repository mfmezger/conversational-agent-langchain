"""The main gui."""
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from langchain.docstore.document import Document
from loguru import logger

from agent.backend.aleph_alpha_service import (
    explain_completion,
    qa_aleph_alpha,
    search_documents_aleph_alpha,
)

# TODO: Background Image


# Constants
PDF_FILE_TYPE = "pdf"
META_DATA_HEIGHT = 500
EXPLANATION_HEIGHT = 300


logger.info("Starting Application.")

# Set small icon in the tab bar
st.set_page_config(page_title="Information Retrieval Embedding Demo", page_icon=":mag:", layout="wide")

# Create title
st.title("Conversational AI")


def create_folder_structure(folder_path: str) -> None:
    """Create the folder structure."""
    Path(folder_path).mkdir(parents=True, exist_ok=True)


def search_documents(token: str, query: str) -> Tuple[str, str, str, List[Tuple[Document, float]]]:
    """Search the documents and return the answer, prompt, and metadata."""
    documents = search_documents_aleph_alpha(query=query, aleph_alpha_token=token)
    answer, prompt, meta_data = qa_aleph_alpha(query=query, documents=documents, aleph_alpha_token=token)
    return answer, prompt, meta_data, documents


def explain(answer: str, prompt: str, token: str) -> str:
    """Explain the answer."""
    explanation = explain_completion(prompt, answer, token)
    return explanation


def initialize() -> None:
    """Initialize the GUI."""
    answer = ""
    prompt = ""

    save_path_input = "data/"
    create_folder_structure(save_path_input)

    # The user needs to enter the aleph alpha api key
    aleph_alpha_api_key = st.text_input("Aleph Alpha Token", type="password")

    st.session_state.api_key = aleph_alpha_api_key
    logger.debug("API Key was entered")

    # Search the documents
    search_query = st.text_input("Search Query")
    if st.button("Start Search", key="start_search"):
        logger.debug("Search was started")
        answer, prompt, meta_data, documents = search_documents(aleph_alpha_api_key, search_query)
        display_results(answer=answer, meta_data=meta_data, documents=documents)

    # Explain the answer
    if st.button("Explain!", key="explain"):
        # make sure that the answer and prompt are not empty
        answer, prompt, meta_data, documents = search_documents(aleph_alpha_api_key, search_query)

        explanation = explain(answer, prompt, aleph_alpha_api_key)
        logger.info(f"Explanation was created{explanation}")
        st.text_area("Explanation", value=explanation, height=EXPLANATION_HEIGHT)


def display_results(answer: str, meta_data: str, documents: str) -> None:
    """Display the search results."""
    st.text_area("QA", value=answer)
    st.text_area("Document", value=documents, height=META_DATA_HEIGHT)
    st.text_area("MetaData", value=meta_data)


# Start the GUI app
initialize()

st.markdown("If you encounter any problems please contact us at: marc.mezger@adesso.de")
