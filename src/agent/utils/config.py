"""Loading the Settings via Pydantic."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Loading the settings with pydantic."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    openai_api_type: str = "openai"
    azure_openai_endpoint: str
    azure_openai_api_key: str
    openai_api_key: str
    openai_api_version: str = "2024-02-15-preview"
    cohere_api_key: str

    # QDRANT
    qdrant_url: str = "http://qdrant"
    qdrant_api_key: str = Field(validation_alias=AliasChoices("qdrant_api_key", "qdrant_cloud_api_key"))
    qdrant_port: int = 6333
    qdrant_prefer_http = False
