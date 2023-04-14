import os

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient

from agent.utils.configuration import load_config

# loguru make file logger
logger.add("logs/{time}.log")


@load_config(location="config/db.yml")
def initialize_qdrant(cfg: DictConfig, aleph_alpha_token: str) -> QdrantClient:
    embeddings = AlephAlphaAsymmetricSemanticEmbedding()
    client = QdrantClient(cfg.db.url, port=cfg.db.port, prefer_grpc=True)
    qdrant = Qdrant(client=client, collection_name="db", embedding_function=embeddings.embed_query)

    return qdrant


def embedd_files(path_to_dir: str, aleph_alpha_token: str):
    loader = DirectoryLoader(path_to_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    logger.info(f"Loaded {len(docs)} documents.")

    # qdrant = initialize_qdrant(aleph_alpha_token=aleph_alpha_token)

    embeddings = AlephAlphaAsymmetricSemanticEmbedding()

    doc_result = embeddings.embed_documents(docs)

    print(doc_result)


# def retrieve_documents(query: str, token: str, store):
#     chain = VectorDBQAWithSourcesChain.from_llm(
#         llm=AlephAlpha(aleph_alpha_api_key=token), vectorstore=store
#     )
#     result = chain({"question": query})

#     return result


if __name__ == "__main__":

    # write the token to the env variable
    os.environ["ALEPH_ALPHA_API_KEY"] = ".."

    embedd_files("data", "adsf")
