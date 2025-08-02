"""Routes to manage embeddings."""

import asyncio
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger
from werkzeug.utils import secure_filename

from agent.backend.services.embedding_management import EmbeddingManagement
from agent.data_model.request_data_model import EmbeddTextRequest
from agent.data_model.response_data_model import EmbeddingResponse
from agent.utils.utility import create_tmp_folder

router = APIRouter()


async def _write_file_to_disk(file_path: Path, file_content: bytes) -> None:
    """Asynchronously writes file content to a specified path using aiofiles.

    This avoids blocking the event loop.
    """
    try:
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_content)
        logger.info(f"Successfully wrote file: {file_path}")
    except OSError as e:
        logger.error(f"Error writing file {file_path}: {e}")
        # Re-raising allows the global exception handler to catch it
        raise


@router.post(
    "/documents",
    tags=["embeddings"],
    summary="Embeds multiple documents from uploaded files.",
)
async def post_embed_documents(
    collection_name: str,
    files: Annotated[list[UploadFile], File(description="A list of files to be embedded.")],
    file_ending: str = ".pdf",
) -> EmbeddingResponse:
    """Endpoint concurrently processes and embeds multiple uploaded documents.

    - **collection_name**: The target collection for the document embeddings.
    - **files**: A list of uploaded files.
    - **file_ending**: The expected file extension for filtering documents.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were uploaded.",
        )

    logger.info(f"Starting embedding process for {len(files)} documents.")
    tmp_dir = create_tmp_folder()
    file_names = []
    write_tasks = []

    for file in files:
        if not file.filename:
            # This case is unlikely with FastAPI's UploadFile but good for safety
            logger.warning("Skipping an upload that had no filename.")
            continue

        file_path = Path(tmp_dir) / file.filename
        file_names.append(file.filename)
        # Create a coroutine to read the file and then write it to disk
        task = asyncio.create_task(_process_and_write_file(file, file_path))
        write_tasks.append(task)

    try:
        await asyncio.gather(*write_tasks)
    except OSError as e:
        # This catches errors from _write_file_to_disk
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="A file writing error occurred on the server.",
        ) from e

    # Consider running this in a thread if it's a blocking CPU-bound operation
    service = EmbeddingManagement(collection_name=collection_name)
    try:
        # This part remains synchronous as per the original code.
        # If embed_documents is I/O bound, it should be made async.
        # If it's CPU-bound, run it in a threadpool.
        await asyncio.to_thread(service.embed_documents, directory=tmp_dir, file_ending=file_ending)
    except Exception as e:
        logger.error(f"Embedding failed for collection '{collection_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to embed the documents.",
        ) from e

    return EmbeddingResponse(status="success", files=file_names)


async def _process_and_write_file(file: UploadFile, file_path: Path) -> None:
    """Reads the content of an uploaded file and writes it to disk."""
    file_content = await file.read()
    await _write_file_to_disk(file_path, file_content)


@router.post("/string/", tags=["embeddings"])
async def embedd_text(embedding: EmbeddTextRequest, collection_name: str) -> EmbeddingResponse:
    """Embedding text."""
    logger.info("Embedding Text")
    service = EmbeddingManagement(collection_name=collection_name)
    tmp_dir = create_tmp_folder()

    sanitized_file_name = secure_filename(embedding.file_name + ".txt")
    full_path = Path(tmp_dir) / sanitized_file_name
    if not str(full_path).startswith(str(tmp_dir)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file name provided.")
    async with aiofiles.open(full_path, "w") as f:
        await f.write(embedding.text)
    await asyncio.to_thread(service.embed_documents, directory=tmp_dir, file_ending=".txt")
    return EmbeddingResponse(status="success", files=[embedding.file_name])
