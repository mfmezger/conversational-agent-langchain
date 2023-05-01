from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "Welcome to the Simple Aleph Alpha FastAPI Backend!"

def test_documents():
    response = client.post("/documents")
    assert response.status_code == 200

def test_search():
    response = client.get("/search")
    assert response.status_code == 200