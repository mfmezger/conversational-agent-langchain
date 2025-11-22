"""The main gui."""

import json
import os
from pathlib import Path

import requests
import streamlit as st
from loguru import logger

# Constants
PDF_FILE_TYPE = "pdf"
META_DATA_HEIGHT = 500
EXPLANATION_HEIGHT = 300

BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
BACKEND_PORT = os.getenv("BACKEND_PORT", "8001")
BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

url_search = f"{BASE_URL}/semantic/search"
url_qa = f"{BASE_URL}/rag/"


logger.info("Starting Application.")

# Set small icon in the tab bar
st.set_page_config(page_title="Conversational Agent", page_icon=":rocket:", layout="wide")

# Create title
st.title("Research Assistant")


def create_folder_structure(folder_path: str) -> None:
    """Create the folder structure."""
    Path(folder_path).mkdir(parents=True, exist_ok=True)


def initialize() -> None:  # noqa: C901
    """Initialize the GUI."""
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # RAG Request
        payload_rag = {"messages": [{"role": "user", "content": prompt}], "collection_name": "default"}
        headers = {"accept": "application/json", "Content-Type": "application/json"}

        # Placeholder for assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            status_container = st.status("Processing...", expanded=True)

            try:
                # Use the streaming endpoint
                response = requests.post(f"{BASE_URL}/rag/stream", json=payload_rag, headers=headers, stream=True, timeout=600)
                response.raise_for_status()

                documents = []

                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        event_type = data.get("type")

                        if event_type == "status":
                            status_container.write(data.get("data"))

                        elif event_type == "content":
                            content = data.get("data")
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")

                        elif event_type == "citation":
                            documents = data.get("data")

                message_placeholder.markdown(full_response)
                status_container.update(label="Finished", state="complete", expanded=False)

                if documents:
                    with st.expander("Show Sources"):
                        for d in documents:
                            st.write("_____")
                            # Handle potential list wrapping in metadata/document
                            source = d["metadata"][0].get("source", "Unknown") if isinstance(d["metadata"], list) else d["metadata"].get("source", "Unknown")
                            page = d["metadata"][0].get("page", "Unknown") if isinstance(d["metadata"], list) else d["metadata"].get("page", "Unknown")
                            text = d["document"][0] if isinstance(d["document"], list) else d["document"]

                            st.markdown(f"**Source:** {source} | **Page:** {page}")
                            st.write(f"Text: {text}")
                            st.write("_____")

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                status_container.update(label="Error", state="error")
                st.error(f"Error communicating with backend: {e}")
                logger.error(f"Backend error: {e}")


# Start the GUI app
initialize()
