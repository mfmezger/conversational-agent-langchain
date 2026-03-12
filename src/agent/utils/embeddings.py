"""Embedding model utilities."""

from langchain_core.embeddings import Embeddings

from agent.utils.config import Config


def get_embedding_model(cfg: Config) -> Embeddings:
    """Return an embeddings client for the configured provider."""
    provider = cfg.embedding_provider
    model_name = cfg.embedding_model_name

    match provider:
        case "cohere":
            from langchain_cohere import CohereEmbeddings  # noqa: PLC0415

            return CohereEmbeddings(model=model_name)

        case "google":
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - depends on optional dependency
                msg = "langchain-google-genai is required for Google embeddings."
                raise ImportError(msg) from exc

            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=cfg.gemini_api_key or None,
                output_dimensionality=cfg.embedding_size,
            )

        case "openai":
            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415

            return OpenAIEmbeddings(
                model=model_name,
                api_key=cfg.openai_api_key or None,
            )

        case _:
            msg = "No suitable embedding Model configured!"
            raise KeyError(msg)
