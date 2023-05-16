"""API Tests."""
import os
import shutil
from io import BytesIO

import httpx
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from agent.api import app, create_tmp_folder

client = TestClient(app)


def test_read_root():
    """Test the root method."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to the Simple Aleph Alpha FastAPI Backend!"


def test_create_tmp_folder():
    """Test the create folder method."""
    tmp_dir = create_tmp_folder()
    assert os.path.exists(tmp_dir)
    shutil.rmtree(tmp_dir)


@pytest.mark.asyncio
async def test_upload_documents():
    """Testing the upload of multiple documents."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        files = [
            UploadFile("file1.txt", file=BytesIO(b"File 1 content")),
            UploadFile("file2.txt", file=BytesIO(b"File 2 content")),
        ]
        response = await ac.post("/embedd_documents", files=[("files", file) for file in files])

    assert response.status_code == 200
    assert response.json() == {
        "message": "Files received and saved.",
        "filenames": ["file1.txt", "file2.txt"],
    }

    # Clean up temporary folders
    for entry in os.scandir():
        if entry.name.startswith("tmp_") and entry.is_dir():
            shutil.rmtree(entry.path)


@pytest.mark.asyncio
async def test_embedd_one_document():
    """Testing the upload of one document."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        tmp_file = UploadFile("file1.txt", file=BytesIO(b"File 1 content"))
        response = await ac.post("/embedd_document/", files=[("file", tmp_file)])

    assert response.status_code == 200
    assert response.json() == {
        "message": "File received and saved.",
        "filenames": "file1.txt",
    }

    # Clean up temporary folders
    for entry in os.scandir():
        if entry.name.startswith("tmp_") and entry.is_dir():
            shutil.rmtree(entry.path)


def test_search():
    """Testing the search."""
    response = client.get("/search?query=test")
    assert response.status_code == 200
