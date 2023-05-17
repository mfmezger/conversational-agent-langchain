"""API Tests."""
import os
import shutil

import httpx
import pytest
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
            open("tests/ressources/1706.03762v5.pdf", "rb"),
            open("tests/ressources/1912.01703v1.pdf", "rb"),
        ]
        response = await ac.post("/embedd_documents", files=[("files", file) for file in files], data={"aa_or_openai": "openai", "token": os.getenv("OPENAI_API_KEY")})

    assert response.status_code == 200
    assert response.json() == {
        "message": "Files received and saved.",
        "filenames": ["1706.03762v5.pdf", "1912.01703v1.pdf"],
    }

    # Clean up temporary folders
    for entry in os.scandir():
        if entry.name.startswith("tmp_") and entry.is_dir():
            shutil.rmtree(entry.path)


@pytest.mark.asyncio
async def test_embedd_one_document():
    """Testing the upload of one document."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        tmp_file = open("tests/ressources/1706.03762v5.pdf", "rb")
        response = await ac.post("/embedd_document/", files=[("file", tmp_file)])

    assert response.status_code == 200
    assert response.json() == {
        "message": "File received and saved.",
        "filenames": "1706.03762v5.pdf",
    }

    # Clean up temporary folders
    for entry in os.scandir():
        if entry.name.startswith("tmp_") and entry.is_dir():
            shutil.rmtree(entry.path)


def test_search_route_invalid_provider():
    """Testing with wrong backend."""
    with pytest.raises(ValueError):
        response = client.get(
            "/search",
            params={
                "query": "example query",
                "aa_or_openai": "invalid_provider",
                "token": "example_token",
                "amount": 3,
            },
        )
        assert response.status_code == 400
        assert "ValueError" in response.text
