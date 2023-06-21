"""The main gui."""
from pathlib import Path
from typing import List, Tuple, Any

import streamlit as st
from loguru import logger

from agent.backend.aleph_alpha_service import (
    embedd_documents_aleph_alpha,
    qa_aleph_alpha,
    search_documents_aleph_alpha,
    explain_completion
)

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


def upload_files(save_path_input: str) -> List[Tuple[str, bytes]]:
    """Upload PDF files and save them to the file system."""
    uploaded_files = st.file_uploader("Upload PDF Files", type=PDF_FILE_TYPE, accept_multiple_files=True)
    files = []
    for file in uploaded_files:
        with open(f"{save_path_input}{file.name}", "wb") as f:
            f.write(file.getbuffer())
        files.append((file.name, file.getbuffer()))
    return files


def start_embedding(file_path: str, token: str) -> None:
    """Start the embedding process."""
    embedd_documents_aleph_alpha(dir=file_path, aleph_alpha_token=token)


def search_documents(token: str, query: str) -> Tuple[str, str, str, Any]:
    """Search the documents and return the answer, prompt, and metadata."""
    documents = search_documents_aleph_alpha(query=query, aleph_alpha_token=token)
    answer, prompt, meta_data = qa_aleph_alpha(query=query, documents=documents, aleph_alpha_token=token)
    logger.info(f"Prompt: {prompt}")
    return answer, prompt, meta_data, documents


def display_results(answer: str, documents: str, meta_data: str) -> None:
    """Display the search results."""
    st.text_area("QA", value=answer)
    st.text_area("Document", value=documents, height=META_DATA_HEIGHT)
    st.text_area("MetaData", value=meta_data)


def explain(answer: str, prompt: str, token: str) -> str:
    """Explain the answer."""
    explanation = explain_completion(prompt, answer, token)
    return explanation


def initialize() -> None:
    """Initialize the GUI."""
    save_path_input = "data/"
    create_folder_structure(save_path_input)

    # The user needs to enter the aleph alpha api key
    aleph_alpha_api_key = st.text_input("Aleph Alpha Token", type="password")
    logger.debug("API Key was entered")

    # Upload PDF files
    files = upload_files(save_path_input)

    # Start the embedding process
    if st.button("Start Embedding"):
        logger.debug("Embedding was started")
        start_embedding(save_path_input, aleph_alpha_api_key)

    # Search the documents
    search_query = st.text_input("Search Query")
    if st.button("Start Search"):
        logger.debug("Search was started")
        answer, prompt, meta_data, documents = search_documents(aleph_alpha_api_key, search_query)
        display_results(answer, meta_data, documents)

    # Explain the answer
    if st.button("Explain!"):
        # Define the answer and prompt variables
        answer = st.text_input("Answer")
        prompt = st.text_input("Prompt")
        explanation = explain(answer, prompt, aleph_alpha_api_key)
        st.text_area("Explanation", value=explanation, height=EXPLANATION_HEIGHT)

    # Search the documents
def display_results(answer: str, meta_data: str, documents: str) -> None:
    """Display the search results."""
    st.text_area("QA", value=answer)
    st.text_area("Document", value=documents, height=META_DATA_HEIGHT)
    st.text_area("MetaData", value=meta_data)

    # Explain the answer
    if st.button("Explain!"):
        explanation = explain(answer, prompt, aleph_alpha_api_key)
        st.text_area("Explanation", value=explanation, height=EXPLANATION_HEIGHT)


# Start the GUI app
initialize()

st.markdown("If you encounter any problems please contact us at: marc.mezger@adesso.de")
