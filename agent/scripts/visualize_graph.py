"""Visualizing the Langgraph Graph."""
from pathlib import Path

from langchain_core.runnables.graph import MermaidDrawMethod

from agent.backend.graph import build_graph

workflow = build_graph()


mermaid_graph = workflow.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

# save as png
with Path("graph.png").open("wb") as f:
    f.write(mermaid_graph)
