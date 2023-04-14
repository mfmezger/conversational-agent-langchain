import os

from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain.vectorstores import Chroma
from loguru import logger
from omegaconf import DictConfig

from agent.utils.configuration import load_config

load_dotenv()


@load_config(location="config/chroma_db.yml")
def get_db_connection_(cfg: DictConfig, aleph_alpha_token: str):
    logger.info(aleph_alpha_token)

    embedding = AlephAlphaAsymmetricSemanticEmbedding()
    vector_db = Chroma(persist_directory=cfg.chroma.persist_directory_aa, embedding_function=embedding)

    return vector_db


@load_config(location="config/chroma_db.yml")
def embedd_documents(cfg: DictConfig, dir: str, aleph_alpha_token: str):
    vector_db = get_db_connection_(aleph_alpha_token=aleph_alpha_token)

    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    logger.info(f"Loaded {len(docs)} documents.")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vector_db.add_texts(texts=texts, metadatas=metadatas)

    vector_db.persist()


def search_documents(aleph_alpha_token: str, query: str):
    vector_db = get_db_connection_(aleph_alpha_token)

    docs = vector_db.similarity_search_with_score(query, k=3)

    return docs


if __name__ == "__main__":
    embedd_documents("data", os.getenv("AA_Token"))

    os.environ["ALEPH_ALPHA_API_KEY"] = os.getenv("AA_Token")

    print(os.getenv("ALEPH_ALPHA_API_KEY"))

    DOCS = search_documents(aleph_alpha_token=os.getenv("AA_token"), query="Muss ich mein Mietwagen volltanken?")
    print(DOCS)
