"""API Tests."""
import os
import shutil

import pytest
from fastapi.testclient import TestClient
from main import app, create_tmp_folder, embedd_documents_wrapper

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


def test_embedd_documents_wrapper_invalid_backend():
    """Invalid Backend tests."""
    with pytest.raises(ValueError):
        embedd_documents_wrapper(folder_name="test_folder", aa_or_openai="invalid_backend")


@pytest.mark.parametrize("aa_or_openai", ["openai", "aleph-alpha"])
def test_upload_documents(aa_or_openai):
    """Test the upload of multiple documents."""
    files = [
        ("file1.txt", "content1"),
        ("file2.txt", "content2"),
    ]
    response = client.post(
        "/embedd_documents",
        params={"aa_or_openai": aa_or_openai},
        files=[("files", (file_name, content)) for file_name, content in files],
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Files received and saved."
    assert sorted(response.json()["filenames"]) == sorted([file_name for file_name, _ in files])


@pytest.mark.parametrize("aa_or_openai", ["openai", "aleph-alpha"])
def test_embedd_one_document(aa_or_openai):
    """Test the upload of only one document."""
    files = ("file1.txt", "content1")
    response = client.post(
        "/embedd_document",
        params={"aa_or_openai": aa_or_openai},
        files=[("file", (files[0], files[1]))],
    )
    assert response.status_code == 200
    assert response.json()["message"] == "File received and saved."
    assert response.json()["filenames"] == files[0]


@pytest.mark.parametrize("aa_or_openai", ["openai", "aleph-alpha"])
def test_search(aa_or_openai):
    """Test the search."""
    query = "example query"
    response = client.get("/search", params={"query": query, "aa_or_openai": aa_or_openai})
    assert response.status_code == 200


