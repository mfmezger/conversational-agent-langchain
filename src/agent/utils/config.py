"""Loading the Settings via Pydantic."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Loading the settings with pydantic."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    openai_api_type: str = "openai"
    azure_openai_endpoint: str
    azure_openai_api_key: str
    openai_api_key: str
    openai_api_version: str = "2024-02-15-preview"
    cohere_api_key: str
    gemini_api_key: str

    # Model Configuration
    model_name: str = "gemini/gemini-2.5-flash"
    embedding_provider: str = "cohere"
    embedding_model_name: str = "embed-v4.0"
    embedding_size: int = 1536

    # QDRANT
    qdrant_url: str = "http://localhost"
    qdrant_api_key: str | None = Field(default=None, validation_alias=AliasChoices("qdrant_api_key", "qdrant_cloud_api_key"))
    qdrant_port: int = 6333
    qdrant_prefer_http: bool = False
    phoenix_collector_endpoint: str = "http://phoenix:4318/v1/traces"
    qdrant_collection_name: str = "default"
    qdrant_embedding_size: int = 1536


config = Config()
