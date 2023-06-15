"""The main gui."""
from pathlib import Path

import streamlit as st
from loguru import logger

from agent.backend.aleph_alpha_service import (
    embedd_documents_aleph_alpha,
    qa_aleph_alpha,
    search_documents_aleph_alpha,
)

logger.info("Starting Aplication.")

# set small icon in the tab bar
st.set_page_config(page_title="Information Retrieval Embedding Demo", page_icon=":mag:")

# create title
st.title("Conversational AI")

# Create a variable to store the selected option
selected_option = st.selectbox("Select an option", ["OpenAI", "Aleph Alpha"], index=1)


def start_embedding(file_path, token):
    """start_embedding starts the embedding process."""
    embedd_documents_aleph_alpha(dir=file_path, aleph_alpha_token=token)


# @load_config("conf/main_conf.yml") cfg: DictConfig
def initialize():
    """Initialize initializes the gui."""
    save_path_input = "data/"

    # create the folder structure
    Path(save_path_input).mkdir(parents=True, exist_ok=True)

    # The user needs to enter the aleph alpha api key
    aleph_alpha_api_key = st.text_input("Aleph Alpha Token", type="password")
    logger.debug("API Key was entered")
    # create a uploader for multiple files of the type pdf
    uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

    # iterate over the Files
    for file in uploaded_files:
        logger.debug("File was uploaded")
        # save the files to the file system in the input folder
        with open(f"{save_path_input}{file.name}", "wb") as f:
            f.write(file.getbuffer())

    # create a button to start the embedding
    if st.button("Start Embedding"):
        logger.debug("Embedding was started")
        start_embedding(save_path_input, aleph_alpha_api_key)

    # create a textfield for the search query
    search_query = st.text_input("Search Query")
    # if the button search is clicked search
    if st.button("Start Search"):
        # search the documents
        logger.debug("Search was started")
        documents = search_documents_aleph_alpha(query=search_query, aleph_alpha_token=aleph_alpha_api_key)
        answer, prompt, meta_data = qa_aleph_alpha(query=search_query, documents=documents, aleph_alpha_token=aleph_alpha_api_key)
        # show the top 3 documents
        st.text_area("QA", value=answer)

        st.text_area("Document", value=documents, height=500)

        st.text_area("MetaData", value=meta_data, height=500)

    if st.button("Explain!"):
        st.text_area("Explanation", value="Not implemented yet", height=500)


# start the gui app
initialize()


st.markdown("If you encounter any problems please contact us at: marc.mezger@adesso.de")
