"""Typed FastAPI dependencies for application-owned resources."""

from typing import Annotated, cast

from fastapi import Depends, Request
from langgraph.graph.state import CompiledStateGraph

from agent.utils.vdb import VDBResources


def get_vdb_resources(request: Request) -> VDBResources:
    """Return the process-scoped vector database resources."""
    return cast("VDBResources", request.app.state.vdb_resources)


def get_graph(request: Request) -> CompiledStateGraph:
    """Return the process-scoped compiled RAG graph."""
    return cast("CompiledStateGraph", request.app.state.graph)


VDBResourcesDep = Annotated[VDBResources, Depends(get_vdb_resources)]
GraphDep = Annotated[CompiledStateGraph, Depends(get_graph)]
