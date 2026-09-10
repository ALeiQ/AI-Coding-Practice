from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_base"

    dense_embedding_model: str = "BAAI/bge-m3"

    ollama_model: str = "qwen2.5"
    ollama_base_url: str = "http://localhost:11434"

    chunk_size: int = 500
    chunk_overlap: int = 75

    top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60


settings = Settings()
