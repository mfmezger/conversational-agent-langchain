"""The main gui."""

import json
from pathlib import Path

import requests
import streamlit as st
from client import AgentClient
from loguru import logger

# Constants
PDF_FILE_TYPE = "pdf"
META_DATA_HEIGHT = 500
EXPLANATION_HEIGHT = 300

client = AgentClient()

logger.info("Starting Application.")

# Set small icon in the tab bar
st.set_page_config(page_title="Conversational Agent", page_icon=":rocket:", layout="wide")

# Create title
st.title("Research Assistant")


def create_folder_structure(folder_path: str) -> None:
    """Create the folder structure."""
    Path(folder_path).mkdir(parents=True, exist_ok=True)


def init_chat_history() -> None:
    """Initialize chat history in session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_chat_history() -> None:
    """Display chat messages from history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_rag_stream(
    response: requests.Response, status_container: st.delta_generator.DeltaGenerator, message_placeholder: st.delta_generator.DeltaGenerator
) -> tuple[str, list]:
    """Process the RAG stream response."""
    full_response = ""
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
    return full_response, documents


def display_sources(documents: list) -> None:
    """Display the sources for the response."""
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


def handle_rag_response(prompt: str) -> None:
    """Handle the RAG response generation and display."""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_container = st.status("Processing...", expanded=True)

        try:
            # Use the streaming endpoint via client
            response = client.chat_stream(messages=[{"role": "user", "content": prompt}], collection_name="default")
            response.raise_for_status()

            full_response, documents = process_rag_stream(response, status_container, message_placeholder)

            message_placeholder.markdown(full_response)
            status_container.update(label="Finished", state="complete", expanded=False)

            display_sources(documents)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except requests.exceptions.HTTPError as err:
            status_container.update(label="Error", state="error")
            error_details = "Unknown error"
            try:
                error_data = err.response.json()
                error_details = error_data.get("details", str(err))
            except ValueError:
                error_details = str(err)

            st.error(f"Backend Error: {error_details}")
            logger.error(f"Backend HTTP error: {err}")

        except Exception as e:
            status_container.update(label="Error", state="error")
            st.error(f"An unexpected error occurred: {e}")
            logger.error(f"Unexpected error: {e}")


def sidebar() -> None:
    """Create the sidebar for document ingestion."""
    with st.sidebar:
        st.header("Document Ingestion")
        collection_name = st.text_input("Collection Name", value="default")
        file_ending = st.selectbox("Document Type", options=[".pdf", ".txt"])

        uploaded_files = st.file_uploader("Choose files", type=["pdf"] if file_ending == ".pdf" else ["txt"], accept_multiple_files=True)

        if st.button("Upload & Embed"):
            if uploaded_files:
                with st.spinner("Uploading and embedding documents..."):
                    try:
                        files = [("files", (file.name, file, file.type)) for file in uploaded_files]

                        response = client.upload_documents(files=files, collection_name=collection_name, file_ending=file_ending)
                        response.raise_for_status()

                        st.success(f"Successfully uploaded {len(uploaded_files)} files!")
                        logger.info(f"Uploaded {len(uploaded_files)} files to collection {collection_name}")

                    except requests.exceptions.HTTPError as err:
                        st.error(f"Error uploading files: {err}")
                        logger.error(f"Upload error: {err}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                        logger.error(f"Unexpected upload error: {e}")
            else:
                st.warning("Please upload at least one file.")


def initialize() -> None:
    """Initialize the GUI."""
    sidebar()
    init_chat_history()
    display_chat_history()

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        handle_rag_response(prompt)


if __name__ == "__main__":
    initialize()
