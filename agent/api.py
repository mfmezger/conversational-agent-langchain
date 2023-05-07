"""FastAPI Backend for the Knowledge Agent."""
import os
import uuid
from typing import List

from fastapi import FastAPI, File, UploadFile
from loguru import logger
from starlette.responses import JSONResponse

from agent.backend.aleph_alpha_service import (
    embedd_documents_aleph_alpha,
    search_documents_aleph_alpha,
)
from agent.backend.openai_service import (
    embedd_documents_openai,
    search_documents_openai,
)

# initialize the Fast API Application.
app = FastAPI(debug=True)


@app.get("/")
def read_root() -> str:
    """Root Message.

    :return: Welcome Message
    :rtype: string
    """
    return "Welcome to the Simple Aleph Alpha FastAPI Backend!"


async def embedd_documents_wrapper(folder_name: str, aa_or_openai: str = "openai", token: str = None):
    """Call the right embedding function for the choosen backend.

    :param folder_name: _description_
    :type folder_name: str
    :param aa_or_openai: _description_, defaults to "openai"
    :type aa_or_openai: str, optional
    :param aleph_alpha_token: _description_, defaults to None
    :type aleph_alpha_token: str, optional
    :raises ValueError: _description_
    """
    if aa_or_openai == "aleph-alpha":
        # Embedd the documents with Aleph Alpha
        embedd_documents_aleph_alpha(dir=folder_name, aleph_alpha_token=token)
    elif aa_or_openai == "openai":
        embedd_documents_openai(dir=folder_name, open_ai_token=token)
        # Embedd the documents with OpenAI#
    else:
        raise ValueError("Please provide either 'aleph-alpha' or 'openai' as a parameter. Other backends are not implemented yet.")


async def create_tmp_folder() -> str:
    """Creates a temporary folder for files to store.

    :return: The directory name
    :rtype: str
    """
    # Create a temporary folder to save the files
    tmp_dir = f"tmp_{str(uuid.uuid4())}"
    os.makedirs(tmp_dir)
    logger.info(f"Created new folder {tmp_dir}.")
    return tmp_dir


@app.post("/embedd_documents")
async def upload_documents(files: List[UploadFile] = File(...), aa_or_openai: str = "openai", token: str = None):
    """Upload multiple documents to the backend.

    :param files: Uploaded files, defaults to File(...)
    :type files: List[UploadFile], optional
    :return: Return as JSON
    :rtype: JSONResponse
    """
    tmp_dir = create_tmp_folder()

    file_names = []

    for file in files:
        file_name = file.filename
        file_names.append(file_name)

        # Save the file to the temporary folder
        with open(os.path.join(tmp_dir, file_name), "wb") as f:
            f.write(await file.read())

    embedd_documents_wrapper(folder_name=tmp_dir, aa_or_openai=aa_or_openai, aleph_alpha_token=token)
    return JSONResponse(content={"message": "Files received and saved.", "filenames": file_names})


@app.post("/embedd_document/")
async def embedd_one_document(file: UploadFile, aa_or_openai: str = "openai", token: str = None):
    """Upload one document to the backend.

    :param file: _description_
    :type file: UploadFile
    :param aa_or_openai: _description_, defaults to "openai"
    :type aa_or_openai: str, optional
    :param aleph_alpha_token: _description_, defaults to None
    :type aleph_alpha_token: str, optional
    :return: _description_
    :rtype: _type_
    """
    # Create a temporary folder to save the files
    tmp_dir = create_tmp_folder()

    with open(os.path.join(tmp_dir, file.file_name), "wb") as f:
        f.write(await file.read())

    embedd_documents_wrapper(folder_name=tmp_dir, aa_or_openai=aa_or_openai, aleph_alpha_token=token)
    return JSONResponse(content={"message": "File received and saved.", "filenames": file.file_name})


@app.get("/search")
def search(query: str, aa_or_openai: str = "openai", token: str = None) -> None:
    """Search for a query in the vector database.

    :param query: The search query
    :type query: str
    :param aa_or_openai: The LLM Provider, defaults to "openai"
    :type aa_or_openai: str, optional
    :param token: Token for the LLM Provider, defaults to None
    :type token: str, optional
    :raises ValueError: If the LLM Provider is not implemented yet
    """
    if aa_or_openai == "aleph-alpha":
        # Embedd the documents with Aleph Alpha
        search_documents_aleph_alpha(aleph_alpha_token=token, query=query)
    elif aa_or_openai == "openai":
        search_documents_openai(open_ai_token=token, query=query)
        # Embedd the documents with OpenAI#
    else:
        raise ValueError("Please provide either 'aleph-alpha' or 'openai' as a parameter. Other backends are not implemented yet.")
