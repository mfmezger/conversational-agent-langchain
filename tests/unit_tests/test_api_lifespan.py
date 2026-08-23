from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from agent.api import lifespan
from agent.utils.config import Config
from agent.utils.vdb import VDBResources


@pytest.mark.anyio
async def test_lifespan_constructs_initializes_and_closes_resources() -> None:
    config = Config()
    sync_client = MagicMock()
    async_client = MagicMock()
    async_client.close = AsyncMock()
    resources = VDBResources(
        config=config,
        sync_client=sync_client,
        async_client=async_client,
        sparse_embeddings=MagicMock(),
    )
    graph = MagicMock()

    with (
        patch("agent.api.Config", return_value=config) as config_cls,
        patch("agent.api.create_vdb_resources", return_value=resources) as create_resources,
        patch("agent.api.initialize_all_vector_dbs", new_callable=AsyncMock) as initialize,
        patch("agent.api.Graph") as graph_cls,
        patch("agent.api.register", return_value=None),
        patch("agent.api.LangChainInstrumentor.instrument", return_value=None),
    ):
        graph_cls.return_value.build_graph.return_value = graph
        app = FastAPI()

        async with lifespan(app):
            assert app.state.vdb_resources is resources
            assert app.state.graph is graph
            sync_client.close.assert_not_called()
            async_client.close.assert_not_awaited()

        config_cls.assert_called_once_with()
        create_resources.assert_called_once_with(config)
        initialize.assert_awaited_once_with(config=config, client=async_client)
        graph_cls.assert_called_once_with(config=config)
        sync_client.close.assert_called_once_with()
        async_client.close.assert_awaited_once_with()


@pytest.mark.anyio
async def test_lifespan_closes_resources_when_startup_fails() -> None:
    sync_client = MagicMock()
    async_client = MagicMock()
    async_client.close = AsyncMock()
    resources = VDBResources(
        config=Config(),
        sync_client=sync_client,
        async_client=async_client,
        sparse_embeddings=MagicMock(),
    )

    with (
        patch("agent.api.Config", return_value=resources.config),
        patch("agent.api.create_vdb_resources", return_value=resources),
        patch("agent.api.initialize_all_vector_dbs", new_callable=AsyncMock, side_effect=RuntimeError("startup failed")),
    ):
        with pytest.raises(RuntimeError, match="startup failed"):
            async with lifespan(FastAPI()):
                pass

    sync_client.close.assert_called_once_with()
    async_client.close.assert_awaited_once_with()
