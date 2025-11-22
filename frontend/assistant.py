"""The main gui."""

from pathlib import Path

import requests
import streamlit as st
from loguru import logger

# Constants
PDF_FILE_TYPE = "pdf"
META_DATA_HEIGHT = 500
EXPLANATION_HEIGHT = 300

url_search = "http://agent:8001/semantic/search"
url_qa = "http://agent:8001/rag/"


logger.info("Starting Application.")

# Set small icon in the tab bar
st.set_page_config(page_title="Conversational Agent", page_icon=":rocket:", layout="wide")

# Create title
st.title("Research Assistant")


def create_folder_structure(folder_path: str) -> None:
    """Create the folder structure."""
    Path(folder_path).mkdir(parents=True, exist_ok=True)


def initialize() -> None:
    """Initialize the GUI."""
    # Search the documents
    search_query = st.text_input("Search Query")
    if st.button("Start Search", key="start_search") and search_query:
        logger.debug("Search was started")

        # RAG Request
        payload_rag = {"messages": [{"role": "user", "content": search_query}], "collection_name": "default"}
        headers = {"accept": "application/json", "Content-Type": "application/json"}

        with st.spinner("Waiting for response...."):
            qa = requests.post(url_qa, json=payload_rag, headers=headers, timeout=6000).json()
            with st.chat_message(name="ai", avatar="🤖"):
                st.write(qa["answer"])

                # Search the documents
                payload_search = {"query": search_query, "k": 5, "collection_name": "default"}
                documents = requests.post(
                    url_search,
                    json=payload_search,
                    timeout=6000,
                ).json()
                # make this one hidden
                # iterate over the objects in the json documents
                for d in documents:
                    with st.expander("Show Results", expanded=False):
                        st.write("_____")
                        col1, col2 = st.columns(2)
                        col2.markdown(f"### Source: {d['source']}")
                        col1.markdown(f"### Page: {d['page']}")

                        st.write(f"Text: {d['text']}")
                        st.write("_____")


# Start the GUI app
initialize()
