from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from config.settings import settings
import logging
import shutil
import os

logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(self):
        azure_embedding = AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        self.embeddings = azure_embedding

    def build_hybrid_retriever(self, docs, session_id: str = "default"):
        logger.info(f"Building hybrid retriever for session {session_id}...")
        
        # Create a unique database path for this session to prevent multi-user conflicts
        db_path = os.path.join(settings.CHROMA_DB_PATH, session_id)
        
        # Clear existing DB for this specific session
        if os.path.exists(db_path):
            logger.info(f"Clearing existing database at {db_path}")
            try:
                shutil.rmtree(db_path)
            except Exception as e:
                logger.warning(f"Could not clear database directory: {e}")

        try:
            # Build vector store
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=db_path,
            )
            logger.info("Chroma retriever built successfully.")
            
            # Build BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriever built successfully.")
            
            vector_retriever = vector_store.as_retriever(
                search_kwargs={"k": settings.VECTOR_SEARCH_K}
            )
            logger.info("Vector retriever built successfully.")
            
            # Build hybrid retriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS,
            )
            logger.info("Hybrid retriever built successfully.")
            
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Error building hybrid retriever: {e}")
            raise