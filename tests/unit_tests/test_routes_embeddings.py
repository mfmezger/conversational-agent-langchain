from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile, status

from agent.data_model.request_data_model import EmbeddTextRequest
from agent.routes.embeddings import _write_file_to_disk, embedd_text, post_embed_documents


pytestmark = pytest.mark.anyio


async def test_write_file_to_disk_success(tmp_path: Path) -> None:
    file_path = tmp_path / "test_file.txt"

    await _write_file_to_disk(file_path, b"test content")

    assert file_path.exists()
    assert file_path.read_bytes() == b"test content"


async def test_write_file_to_disk_error() -> None:
    with pytest.raises(OSError):
        await _write_file_to_disk(Path("/"), b"content")


@patch("agent.routes.embeddings.create_tmp_folder")
@patch("agent.routes.embeddings.EmbeddingManagement")
@patch("agent.routes.embeddings._write_file_to_disk")
async def test_post_embed_documents_success(mock_write, mock_service_cls, mock_tmp_folder, tmp_path: Path) -> None:
    mock_tmp_folder.return_value = str(tmp_path)
    mock_service_cls.return_value = MagicMock()

    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.pdf"
    mock_file.read = AsyncMock(return_value=b"content")

    response = await post_embed_documents(collection_name="test_collection", files=[mock_file], file_ending=".pdf")

    assert response.status == "success"
    assert response.files == ["test.pdf"]
    mock_write.assert_called_once()
    mock_service_cls.return_value.embed_documents.assert_called_once()


async def test_post_embed_documents_no_files() -> None:
    with pytest.raises(HTTPException) as exc:
        await post_embed_documents(collection_name="test", files=[])

    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc.value.detail == "No files were uploaded."


@patch("agent.routes.embeddings.create_tmp_folder")
@patch("agent.routes.embeddings._write_file_to_disk")
async def test_post_embed_documents_write_error(mock_write, mock_tmp_folder, tmp_path: Path) -> None:
    mock_tmp_folder.return_value = str(tmp_path)
    mock_write.side_effect = OSError("Write failed")

    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.pdf"
    mock_file.read = AsyncMock(return_value=b"content")

    with pytest.raises(HTTPException) as exc:
        await post_embed_documents(collection_name="test", files=[mock_file])

    assert exc.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "file writing error" in exc.value.detail


@patch("agent.routes.embeddings.create_tmp_folder")
@patch("agent.routes.embeddings.EmbeddingManagement")
@patch("agent.routes.embeddings._write_file_to_disk")
async def test_post_embed_documents_embedding_error(mock_write, mock_service_cls, mock_tmp_folder, tmp_path: Path) -> None:
    mock_tmp_folder.return_value = str(tmp_path)
    mock_service = MagicMock()
    mock_service.embed_documents.side_effect = Exception("Embedding failed")
    mock_service_cls.return_value = mock_service

    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.pdf"
    mock_file.read = AsyncMock(return_value=b"content")

    with pytest.raises(HTTPException) as exc:
        await post_embed_documents(collection_name="test", files=[mock_file])

    assert exc.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to embed" in exc.value.detail


@patch("agent.routes.embeddings.create_tmp_folder")
@patch("agent.routes.embeddings.EmbeddingManagement")
@patch("aiofiles.open")
async def test_embedd_text_success(mock_aio_open, mock_service_cls, mock_tmp_folder, tmp_path: Path) -> None:
    mock_tmp_folder.return_value = str(tmp_path)
    mock_service_cls.return_value = MagicMock()

    mock_file = AsyncMock()
    mock_aio_open.return_value.__aenter__.return_value = mock_file

    request = EmbeddTextRequest(text="some text", file_name="test_doc")

    response = await embedd_text(embedding=request, collection_name="test_collection")

    assert response.status == "success"
    assert response.files == ["test_doc"]
    mock_file.write.assert_called_once_with("some text")
    mock_service_cls.return_value.embed_documents.assert_called_once()
