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
url_qa = "http://agent:8001/qa"


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
    # The user needs to enter the aleph alpha api key
    aleph_alpha_api_key = st.text_input("Aleph Alpha Token", type="password")

    st.session_state.api_key = aleph_alpha_api_key
    logger.debug("API Key was entered")

    # Search the documents
    search_query = st.text_input("Search Query")
    if st.button("Start Search", key="start_search") and search_query:
        logger.debug("Search was started")

        params = {"query": search_query, "llm_backend": "aa", "token": st.session_state.api_key, "amount": "1"}
        headers = {"accept": "application/json"}

        with st.spinner("Waiting for response...."):
            qa = requests.post(url_qa, params=params, headers=headers).json()
            with st.chat_message(name="ai", avatar="🤖"):
                st.write(qa["answer"])

                # Search the documents
                documents = requests.post(
                    url_search,
                    json={
                        "search": {
                            "query": search_query,
                            "llm_backend": {
                                "llm_provider": "aa",
                                "token": st.session_state.api_key,
                            },
                            "filtering": {"threshold": 0, "collection_name": "aleph-alpha", "filter": {}},
                            "amount": 5,
                        },
                        "language": "detect",
                        "history": 0,
                        "history_list": [],
                    },
                ).json()
                # make this one hidden
                # iterate over the objects in the json documents
                for d in documents:
                    with st.expander("Show Results", expanded=False):
                        st.write("_____")
                        col1, col2, col3 = st.columns(3)
                        col3.markdown(f"### Source: {d['source']}")
                        col1.markdown(f"### Page: {d['page']}")
                        col2.markdown(f"### Score: {d['score']}")

                        st.write(f"Text: {d['text']}")
                        st.write("_____")


# Start the GUI app
initialize()
