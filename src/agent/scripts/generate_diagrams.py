"""Generate a diagram of the architecture."""

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.onprem.client import Users
from diagrams.programming.flowchart import Database
from diagrams.programming.framework import FastAPI

graph_attr = {"fontsize": "45", "bgcolor": "white"}
node_attr = {"fontsize": "20"}
edge_attr = {"fontsize": "32"}

resources_dir = Path(__file__).parent / "resources"

with Diagram(
    "Conversational Agent Diagram",
    show=False,
    graph_attr=graph_attr,
    edge_attr=edge_attr,
    node_attr=node_attr,
    outformat="png",
    filename=str(resources_dir / "Architecture"),
):
    with Cluster("Local PC / Cloud"):
        with Cluster("Frontend"):
            streamlit = Custom("Streamlit", str(resources_dir / "streamlit.png"))

        with Cluster("Backend Services"):
            fastapi = FastAPI("FastAPI Backend")
            langgraph = Custom("LangGraph", str(resources_dir / "langchain.png"))
            qdrant = Database("Qdrant Vector DB")
            phoenix = Custom("Arize Phoenix", str(resources_dir / "phoenix.png"))

    with Cluster("External LLM APIs"):
        cohere = Custom("Cohere", str(resources_dir / "cohere.png"))
        openai = Custom("OpenAI", str(resources_dir / "openai.png"))
        gemini = Custom("Google Gemini", str(resources_dir / "gemini.png"))

    users = Users("Users")

    # User interactions
    users >> Edge(label="Interacts with", style="bold") >> streamlit
    streamlit >> Edge(label="API Requests", style="bold") >> fastapi

    # Backend flow
    fastapi >> Edge(label="Orchestrates", style="bold") >> langgraph
    langgraph >> Edge(label="Retrieves Context", style="bold") >> qdrant

    # LLM Calls
    langgraph >> Edge(label="Generates/Embeds", style="bold") >> cohere
    langgraph >> Edge(label="Generates", style="bold") >> openai
    langgraph >> Edge(label="Generates", style="bold") >> gemini

    # Observability
    fastapi >> Edge(label="Traces", style="dashed") >> phoenix
    langgraph >> Edge(label="Traces", style="dashed") >> phoenix
