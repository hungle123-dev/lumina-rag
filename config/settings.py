from pydantic_settings import BaseSettings
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES
import os

class Settings(BaseSettings):
    # Azure OpenAI settings
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str = "2023-05-15"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "my-embedding-model"

    # OpenRouter settings (replaces Groq)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    RESEARCH_MODEL: str = "deepseek/deepseek-chat-v3-0324"  # Very cheap & powerful
    VERIFICATION_MODEL: str = "deepseek/deepseek-chat-v3-0324"
    LLAMA_CLOUD_API_KEY: str = ""  # For LlamaParse cloud parser

    # Optional settings with defaults
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list = ALLOWED_TYPES

    # Database settings
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    # Retrieval settings
    VECTOR_SEARCH_K: int = 6
    HYBRID_RETRIEVER_WEIGHTS: list = [0.4, 0.6]

    # Logging settings
    LOG_LEVEL: str = "INFO"

    # New cache settings with type annotations
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow old .env keys like GROQ_API_KEY without crashing

settings = Settings()