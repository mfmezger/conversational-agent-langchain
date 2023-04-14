"""FastAPI Backend for the Knowledge Agent."""
from fastapi import FastAPI

# initialize the Fast API Application.
app = FastAPI(debug=True)


@app.get("/")
def read_root():
    """Root Message.

    :return: Welcome Message
    :rtype: string
    """
    return "Welcome to the Simple Aleph Alpha FastAPI Backend!"


@app.post("/documents")
def documents():
    """_summary_."""
    pass


@app.get("/search")
def search():
    """_summary_."""
    pass
