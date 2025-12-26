"""The main gui."""

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
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


def display_sources(documents: list[dict]) -> None:
    """Display the sources."""
    if documents:
        with st.expander("Sources"):
            for doc in documents:
                st.markdown(f"**Page {doc.get('page', 'Unknown')}**")
                st.markdown(doc.get("text", ""))
                st.markdown("---")


async def process_rag_stream(
    stream: AsyncGenerator[str, None],
    status_container: st.delta_generator.DeltaGenerator,
    message_placeholder: st.delta_generator.DeltaGenerator,
) -> tuple[str, list]:
    """Process the RAG stream response."""
    full_response = ""
    documents = []

    async for line in stream:
        if line:
            data = json.loads(line)
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


async def handle_rag_response_async(prompt: str) -> None:
    """Handle the RAG response generation and display asynchronously."""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_container = st.status("Processing...", expanded=True)

        try:
            # Use the streaming endpoint via client
            stream = client.chat_stream(messages=[{"role": "user", "content": prompt}], collection_name="default")

            full_response, documents = await process_rag_stream(stream, status_container, message_placeholder)

            message_placeholder.markdown(full_response)
            status_container.update(label="Finished", state="complete", expanded=False)

            display_sources(documents)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except httpx.HTTPStatusError as err:
            status_container.update(label="Error", state="error")
            error_details = "Unknown error"
            try:
                # Try to read response text if available, though for stream it might be hard
                error_details = str(err)
            except ValueError:
                error_details = str(err)

            st.error(f"Backend Error: {error_details}")
            logger.error(f"Backend HTTP error: {err}")

        except Exception as e:
            status_container.update(label="Error", state="error")
            st.error(f"An unexpected error occurred: {e}")
            logger.error(f"Unexpected error: {e}")


def handle_rag_response(prompt: str) -> None:
    """Wrapper to run async handler."""
    asyncio.run(handle_rag_response_async(prompt))


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

                        # Run async upload
                        response = asyncio.run(client.upload_documents(files=files, collection_name=collection_name, file_ending=file_ending))
                        response.raise_for_status()

                        st.success(f"Successfully uploaded {len(uploaded_files)} files!")
                        logger.info(f"Uploaded {len(uploaded_files)} files to collection {collection_name}")

                    except httpx.HTTPStatusError as err:
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
