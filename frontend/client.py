"""Client for interacting with the Agent API."""

import os

import requests


class AgentClient:
    """Client for interacting with the Agent API."""

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize the client."""
        if base_url:
            self.base_url = base_url
        else:
            host = os.getenv("BACKEND_HOST", "localhost")
            port = os.getenv("BACKEND_PORT", "8001")
            self.base_url = f"http://{host}:{port}"

    def chat_stream(self, messages: list[dict], collection_name: str = "default") -> requests.Response:
        """Stream chat response from the backend."""
        payload = {"messages": messages, "collection_name": collection_name}
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        return requests.post(f"{self.base_url}/rag/stream", json=payload, headers=headers, stream=True, timeout=600)

    def upload_documents(self, files: list[tuple], collection_name: str = "default", file_ending: str = ".pdf") -> requests.Response:
        """Upload documents to the backend."""
        params = {"collection_name": collection_name, "file_ending": file_ending}
        return requests.post(f"{self.base_url}/embeddings/documents", params=params, files=files, timeout=6000)

    def search(self, query: str, collection_name: str = "default") -> requests.Response:
        """Search for documents."""
        params = {"query": query, "collection_name": collection_name}
        return requests.get(f"{self.base_url}/semantic/search", params=params, timeout=60)
