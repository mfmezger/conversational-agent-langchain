"""Client for interacting with the Agent API."""

import os
from collections.abc import AsyncGenerator

import httpx


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

    async def chat_stream(self, messages: list[dict], collection_name: str = "default") -> AsyncGenerator[str, None]:
        """Stream chat response from the backend."""
        payload = {"messages": messages, "collection_name": collection_name}
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=600.0) as client, client.stream("POST", f"{self.base_url}/rag/stream", json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line

    async def upload_documents(self, files: list[tuple], collection_name: str = "default", file_ending: str = ".pdf") -> httpx.Response:
        """Upload documents to the backend."""
        params = {"collection_name": collection_name, "file_ending": file_ending}
        # httpx handles files differently than requests
        # files list of tuples (name, (filename, content, content_type))
        async with httpx.AsyncClient(timeout=6000.0) as client:
            return await client.post(f"{self.base_url}/embeddings/documents", params=params, files=files)

    async def search(self, query: str, collection_name: str = "default") -> httpx.Response:
        """Search for documents."""
        params = {"query": query, "collection_name": collection_name}
        async with httpx.AsyncClient(timeout=60.0) as client:
            return await client.get(f"{self.base_url}/semantic/search", params=params)
