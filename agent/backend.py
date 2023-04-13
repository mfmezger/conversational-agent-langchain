from langchain.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain.vectorstores import Qdrant
from langchain.document_loaders import DirectoryLoader
from utils.configuration import load_config
from omegaconf import DictConfig
from loguru import logger
from qdrant_client import QdrantClient
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import VectorDBQAWithSourcesChain
from langchain import AlephAlpha


@load_config(location="config/db.yml")
def initialize_qdrant(cfg: DictConfig) -> QdrantClient:
    client = QdrantClient()
    collection_name = "DB"
    embedding_function = AlephAlphaAsymmetricSemanticEmbedding()
    qdrant = Qdrant(client, collection_name, embedding_function)
    return qdrant

def embedd_files(path_to_dir: str, token: str):

    loader = DirectoryLoader(path_to_dir)

    index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Qdrant, 
    embedding=AlephAlphaAsymmetricSemanticEmbedding(aleph_alpha_api_key=token),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    )

    index = index_creator.from_loaders([loader])
    retriever = index.vectorstore.as_retriever()

    # embedd the documents using Aleph Alpha
    # embeddings = AlephAlphaAsymmetricSemanticEmbedding()
    # embeddings = embeddings.embed_documents(docs)
    logger.info("SUCCESS: Documents embedded!")
    # store the embeddings in Qdrant
    # qdrant = Qdrant.from_documents(docs, embeddings, host=cfg.db.host, prefer_grpc=True, api_key=cfg.db.api_key)
    logger.info("SUCCESS: Embeddings stored in DB!")


    return index

def retrieve_documents(query: str, token: str, store):

    chain = VectorDBQAWithSourcesChain.from_llm(llm=AlephAlpha(aleph_alpha_api_key=token), vectorstore=store)
    result = chain({"question": query})

    return result






