import pytest
import os
import requests
from fastapi.testclient import TestClient
from agent.api import app

client = TestClient(app)

@pytest.mark.parametrize("llm_provider", ["openai", "cohere", "ollama"])
def test_create_collection(llm_provider: str):
    response = client.post(f"/collection/create/{llm_provider}/test")

    assert response.status_code == 200
    assert response.json == {"message": "Collection test created."}

    # cleanup
    response = requests.delete(f"http://localhost:6333/collection/delete/test", headers={"api_key": os.getenv("QDRANT_API_KEY")})



def test_invalid_provider():
    response = client.post(f"/collection/create/test/test")

    assert response.status_code == 500
