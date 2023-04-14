"""FastAPI Backend for the Knowledge Agent."""
from fastapi import FastAPI

# from loguru import logger

# initialize the Fast API Application.
app = FastAPI(debug=True)


@app.get("/")
def read_root() -> str:
    """Root Message.

    :return: Welcome Message
    :rtype: string
    """
    return "Welcome to the Simple Aleph Alpha FastAPI Backend!"


@app.post("/documents")
def documents() -> None:
    """_summary_."""
    pass


@app.get("/search")
def search() -> None:
    """_summary_."""
    pass
