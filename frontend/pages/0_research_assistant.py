"""The main gui."""
from pathlib import Path

import requests
import streamlit as st
from loguru import logger

# Constants
PDF_FILE_TYPE = "pdf"
META_DATA_HEIGHT = 500
EXPLANATION_HEIGHT = 300

url = "http://localhost:8001/semantic/search"


logger.info("Starting Application.")

# Set small icon in the tab bar
st.set_page_config(page_title="Information Retrieval Embedding Demo", page_icon=":mag:", layout="wide")

# Create title
st.title("Conversational AI")


def create_folder_structure(folder_path: str) -> None:
    """Create the folder structure."""
    Path(folder_path).mkdir(parents=True, exist_ok=True)


def initialize() -> None:
    """Initialize the GUI."""
    answer = ""
    prompt = ""

    # The user needs to enter the aleph alpha api key
    aleph_alpha_api_key = st.text_input("Aleph Alpha Token", type="password")

    st.session_state.api_key = aleph_alpha_api_key
    logger.debug("API Key was entered")

    # Search the documents
    search_query = st.text_input("Search Query")
    if st.button("Start Search", key="start_search") and search_query:
        logger.debug("Search was started")

        # Search the documents
        documents = requests.post(
            url,
            json={
                "query": search_query,
                "llm_backend": "aa",
                "token": st.session_state.api_key,
                "amount": 5,
            },
        ).json()

        # iterate over the objects in the json documents


# Start the GUI app
initialize()

st.markdown("If you encounter any problems please contact us at: marc.mezger@adesso.de")
