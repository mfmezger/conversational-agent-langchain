import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import UploadFile, HTTPException, status
from pathlib import Path
import shutil
import tempfile
import os
from agent.routes.embeddings import _write_file_to_disk, post_embed_documents, embedd_text
from agent.data_model.request_data_model import EmbedTextRequest

class TestEmbeddingsRoutes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    async def asyncTearDown(self):
        shutil.rmtree(self.test_dir)

    async def test_write_file_to_disk_success(self):
        file_path = self.test_path / "test_file.txt"
        content = b"test content"

        await _write_file_to_disk(file_path, content)

        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.read_bytes(), content)

    async def test_write_file_to_disk_error(self):
        # Try to write to a directory path which should raise OSError
        with self.assertRaises(OSError):
            await _write_file_to_disk(Path("/"), b"content")

    @patch("agent.routes.embeddings.create_tmp_folder")
    @patch("agent.routes.embeddings.EmbeddingManagement")
    @patch("agent.routes.embeddings._write_file_to_disk")
    async def test_post_embed_documents_success(self, mock_write, mock_service_cls, mock_tmp_folder):
        # Setup mocks
        mock_tmp_folder.return_value = self.test_dir
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.pdf"
        mock_file.read = AsyncMock(return_value=b"content")

        # Call endpoint
        response = await post_embed_documents(
            collection_name="test_collection",
            files=[mock_file],
            file_ending=".pdf"
        )

        # Assertions
        self.assertEqual(response.status, "success")
        self.assertEqual(response.files, ["test.pdf"])
        mock_write.assert_called_once()
        mock_service.embed_documents.assert_called_once()

    async def test_post_embed_documents_no_files(self):
        with self.assertRaises(HTTPException) as cm:
            await post_embed_documents(collection_name="test", files=[])

        self.assertEqual(cm.exception.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(cm.exception.detail, "No files were uploaded.")

    @patch("agent.routes.embeddings.create_tmp_folder")
    @patch("agent.routes.embeddings._write_file_to_disk")
    async def test_post_embed_documents_write_error(self, mock_write, mock_tmp_folder):
        mock_tmp_folder.return_value = self.test_dir
        mock_write.side_effect = OSError("Write failed")

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.pdf"
        mock_file.read = AsyncMock(return_value=b"content")

        with self.assertRaises(HTTPException) as cm:
            await post_embed_documents(collection_name="test", files=[mock_file])

        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("file writing error", cm.exception.detail)

    @patch("agent.routes.embeddings.create_tmp_folder")
    @patch("agent.routes.embeddings.EmbeddingManagement")
    @patch("agent.routes.embeddings._write_file_to_disk")
    async def test_post_embed_documents_embedding_error(self, mock_write, mock_service_cls, mock_tmp_folder):
        mock_tmp_folder.return_value = self.test_dir
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service
        mock_service.embed_documents.side_effect = Exception("Embedding failed")

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.pdf"
        mock_file.read = AsyncMock(return_value=b"content")

        with self.assertRaises(HTTPException) as cm:
            await post_embed_documents(collection_name="test", files=[mock_file])

        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Failed to embed", cm.exception.detail)

    @patch("agent.routes.embeddings.create_tmp_folder")
    @patch("agent.routes.embeddings.EmbeddingManagement")
    @patch("aiofiles.open")
    async def test_embedd_text_success(self, mock_aio_open, mock_service_cls, mock_tmp_folder):
        mock_tmp_folder.return_value = self.test_dir
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        # Mock aiofiles context manager
        mock_file = AsyncMock()
        mock_aio_open.return_value.__aenter__.return_value = mock_file

        request = EmbedTextRequest(text="some text", file_name="test_doc")

        response = await embedd_text(embedding=request, collection_name="test_collection")

        self.assertEqual(response.status, "success")
        self.assertEqual(response.files, ["test_doc"])
        mock_file.write.assert_called_once_with("some text")
        mock_service.embed_documents.assert_called_once()
