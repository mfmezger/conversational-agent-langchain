"""Script that contains the Pydantic Models for the Rest Request."""
from typing import List, Optional

from fastapi import UploadFile
from pydantic import BaseModel, Field


class QARequest(BaseModel):
    """Request for the QA endpoint."""

    query: Optional[str] = Field(None, title="Query", description="The question to answer.")
    llm_backend: str = Field("aa", title="LLM Provider", description="The LLM provider to use for answering the question. Can be 'openai' or 'aleph-alpha'.")
    token: Optional[str] = Field(None, title="API Token", description="The API token for the LLM provider.")
    collection_name: Optional[str] = Field(None, title="Name of the Collection", description="Name of the Qdrant Collection.")
    amount: int = Field(1, title="Amount", description="The number of answers to return.")
    language: str = Field("de", title="Language", description="The language to use for the answer.")
    history: int = Field(0, title="History", description="The number of previous questions to include in the context.")
    history_list: List[str] = Field(None, title="History List", description="A list of previous questions to include in the context.")


class EmbeddTextFilesRequest(BaseModel):
    """The request for the Embedd Text Files endpoint."""

    files: List[UploadFile] = Field(..., description="The list of text files to embed.")
    llm_backend: str = Field("aa", description="The LLM provider to use for embedding.")
    token: Optional[str] = Field(None, description="The API token for the LLM provider.")
    seperator: str = Field("###", description="The seperator to use between embedded texts.")


class SearchRequest(BaseModel):
    """The request parameters for searching the database."""

    query: str = Field(..., title="Query", description="The search query.")
    llm_backend: str = Field("aa", title="LLM Provider", description="The LLM provider to use for searching.")
    collection_name: Optional[str] = Field(None, title="Name of the Collection", description="Name of the Qdrant Collection.")
    token: Optional[str] = Field(None, title="API Token", description="The API token for the LLM provider.")
    amount: int = Field(3, title="Amount", description="The number of search results to return.")


class EmbeddTextRequest(BaseModel):
    """The request parameters for embedding text."""

    text: str = Field(..., title="Text", description="The text to embed.")
    file_name: str = Field(..., title="File Name", description="The name of the file to save the embedded text to.")
    llm_backend: str = Field("aa", title="LLM Provider", description="The LLM provider to use for embedding.")
    token: Optional[str] = Field(None, title="API Token", description="The API token for the LLM provider.")
    seperator: str = Field("###", title="seperator", description="The seperator to use between embedded texts.")


class ExplainRequest(BaseModel):
    """The request parameters for explaining the output."""

    prompt: str = Field(..., title="Prompt", description="The prompt used to generate the output.")
    collection_name: Optional[str] = Field(None, title="Name of the Collection", description="Name of the Qdrant Collection.")
    output: str = Field(..., title="Output", description="The output to be explained.")
    token: Optional[str] = Field(None, title="API Token", description="The Aleph Alpha API token.")
    llm_backend: str = Field("aa", title="LLM Provider", description="The LLM provider to use for embedding.")


class CustomPromptCompletion(BaseModel):
    """The Custom Prompt Completion Model."""

    token: str = Field(..., title="Token", description="The API token for the LLM provider.")
    prompt: str = Field(..., title="Prompt", description="The prompt to use for the completion.")
    llm_backend: str = Field(
        "aa", title="LLM Provider", description="The LLM provider to use for embedding. Can be 'aa' for Aleph Alpha or 'openai' for OpenAI, or 'gpt4allL'."
    )
    model: str = Field(..., title="Model", description="The model to use for the completion.")
    max_tokens: int = Field(256, title="Max Tokens", description="The maximum number of tokens to generate.")
    temperature: float = Field(..., title="Temperature", description="The temperature to use for the completion.")
    stop_sequences: List[str] = Field(..., title="Stop Sequences", description="The stop sequences to use for the completion.")
