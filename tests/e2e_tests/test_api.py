"""API Tests."""
import os
import shutil

import httpx
import pytest
from fastapi.testclient import TestClient
from loguru import logger

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
@pytest.mark.parametrize("provider", ["openai", "aleph-alpha"])
async def test_upload_documents(provider):
    """Testing the upload of multiple documents."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        files = [
            open("tests/resources/1706.03762v5.pdf", "rb"),
            open("tests/resources/1912.01703v1.pdf", "rb"),
        ]
        if provider == "openai":
            logger.warning("Using OpenAI API")
            response = await ac.post(
                "/embedd_documents", params={"llm_backend": "openai", "token": os.getenv("OPENAI_API_KEY")}, files=[("files", file) for file in files]
            )
        else:
            logger.warning("Using Aleph Alpha API")
            response = await ac.post(
                "/embedd_documents", params={"llm_backend": "aleph-alpha", "token": os.getenv("ALEPH_ALPHA_API_KEY")}, files=[("files", file) for file in files]
            )

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
        tmp_file = open("tests/resources/1706.03762v5.pdf", "rb")
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
        response = client.post(
            "/search",
            json={
                "query": "example query",
                "llm_backend": "invalid_provider",
                "token": "example_token",
                "amount": 3,
            },
        )
        assert response.status_code == 400
        assert "ValueError" in response.text


def test_search_route():
    """Testing with wrong backend."""
    response = client.post(
        "/search",
        json={
            "query": "Was ist Vanilin?",
            "llm_backend": "aa",
            "amount": 3,
        },
    )
    assert response.status_code == 200
    assert response.json() is not None


def test_explain_output():
    """Test the function with valid arguments."""
    response = client.post("/explain", json={"prompt": "What is the capital of France?", "output": "Paris", "token": os.getenv("ALEPH_ALPHA_API_KEY")})
    assert response.status_code == 200


def test_wrong_input_explain_output():
    """Test the function with wrong arguments."""
    with pytest.raises(ValueError):
        client.post("/explain", json={"prompt": "", "output": "", "token": os.getenv("ALEPH_ALPHA_API_KEY")})
    with pytest.raises(ValueError):
        client.post("/explain", json={"prompt": "", "output": "asdfasdf", "token": os.getenv("ALEPH_ALPHA_API_KEY")})
    with pytest.raises(ValueError):
        client.post("/explain", json={"prompt": "asdfasdf", "output": "", "token": os.getenv("ALEPH_ALPHA_API_KEY")})


def test_embedd_text():
    """Test the embedd_text function."""
    # load text
    with open("tests/resources/file1.txt") as f:
        text = f.read()

    response = client.post("/embedd_text", json={"text": text, "llm_backend": "aa", "file_name": "file", "token": os.getenv("ALEPH_ALPHA_API_KEY"), "seperator": "###"})
    logger.info(response)
    assert response.status_code == 200
    logger.info(response.json())
    assert response.json() == {"message": "Text received and saved.", "filenames": "file"}
