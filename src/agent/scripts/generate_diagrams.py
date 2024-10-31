"""Generate a diagram of the architecture."""

from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.onprem.client import Client, Users
from diagrams.programming.flowchart import Database
from diagrams.programming.framework import FastAPI

graph_attr = {"fontsize": "45", "bgcolor": "white"}
node_attr = {"fontsize": "20"}
edge_attr = {"fontsize": "32"}

with Diagram(
    "Conversational Agent Diagram", show=False, graph_attr=graph_attr, edge_attr=edge_attr, node_attr=node_attr, outformat="png", filename="resources/Architecture"
):
    with Cluster("Local PC/Docker/Ollama Server"):
        ollama = Custom("Ollama", "../src/agent/scripts/resources/ollama.png")
    with Cluster("Database Docker Container"):
        db = Database("Qdrant")

    with Cluster("Conversational Backend  Docker Container"):
        fastapi = FastAPI("Rest API")

        fastapi >> Edge(label="API Call", reverse=True, style="bold") >> ollama
        fastapi >> Edge(label="retrieve Documents", reverse=True, style="bold") >> db

    with Cluster("Frontend"):
        client = Client("Frontend")

    with Cluster("LLM APIs"):
        cohere = Custom("Cohere", "../src/agent/scripts/resources/cohere.png")
        openai = Custom("OpenAI", "../src/agent/scripts/resources/openai.png")

    users = Users("Users")

    users >> Edge(label="Access Frontend", reverse=True, style="bold") >> client >> Edge(label="API Call", reverse=True, style="bold") >> fastapi
    users >> Edge(label="API Call", reverse=True, style="bold") >> fastapi

    fastapi >> Edge(label="API Call", reverse=True, style="bold") >> cohere
    fastapi >> Edge(label="API Call", reverse=True, style="bold") >> openai
