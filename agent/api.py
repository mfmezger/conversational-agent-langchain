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


# from enum import Enum
# from pydantic import BaseModel, validator


# class Profession(str, Enum):
#    DS = "data scientist"
#    MLE = "machine learning scientist"
#    RS = "research scientist"

# class NewHire(BaseModel):
#     profession: Profession
#     name: str
    
#     @validator('name')
#     def name_must_contain_space(cls, v):
#         if ' ' not in v:
#             raise ValueError('Name must contain a space for first and last name.')
#         return v