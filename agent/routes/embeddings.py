"""Routes to manage embeddings."""
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from loguru import logger

from agent.backend.LLMStrategy import LLMContext, LLMStrategyFactory
from agent.data_model.request_data_model import EmbeddTextRequest, LLMBackend
from agent.data_model.response_data_model import EmbeddingResponse
from agent.utils.utility import create_tmp_folder

router = APIRouter()


@router.post("/documents", tags=["embeddings"])
async def post_embedd_documents(llm_backend: LLMBackend, files: list[UploadFile] = File(...), file_ending: str = ".pdf") -> EmbeddingResponse:
    """Embeds multiple documents from files.

    Args:
    ----
        llm_backend (LLMBackend): Which LLM backend to use.
        files (list[UploadFile], optional): The Uploaded files .Defaults to File(...).
        file_ending (str, optional): The file ending of the uploaded file. Defaults to ".pdf".

    Raises:
    ------
        ValueError: _description_
        ValueError: _description_

    Returns:
    -------
        EmbeddingResponse: _description_
    """
    logger.info("Embedding Multiple Documents")
    tmp_dir = create_tmp_folder()

    service = LLMContext(LLMStrategyFactory.get_strategy(strategy_type=LLMProvider.ALEPH_ALPHA, collection_name=llm_backend.collection_name))
    file_names = []

    for file in files:
        file_name = file.filename
        file_names.append(file_name)
        if tmp_dir is None or not Path(tmp_dir).exists():
            msg = "Please provide a temporary folder to save the files."
            raise ValueError(msg)
        if file_name is None:
            msg = "Please provide a file to save."
            raise ValueError(msg)
        with Path(tmp_dir / file_name).open("wb") as f:
            f.write(await file.read())

    service.embed_documents(folder=tmp_dir, file_ending=file_ending)
    return EmbeddingResponse(status="success", files=file_names)


@router.post("/string/", tags=["embeddings"])
async def embedd_text(embedding: EmbeddTextRequest, llm_backend: LLMBackend) -> EmbeddingResponse:
    """Embedding text."""
    logger.info("Embedding Text")
    service = LLMContext(LLMStrategyFactory.get_strategy(strategy_type=llm_backend.llm_provider, collection_name=llm_backend.collection_name))
    tmp_dir = create_tmp_folder()
    with (Path(tmp_dir) / (embedding.file_name + ".txt")).open("w") as f:
        f.write(embedding.text)
    service.embed_documents(directory=tmp_dir, file_ending=".txt")
    return EmbeddingResponse(status="success", files=[embedding.file_name])
