from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。所有外部服务地址都从环境变量注入。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "zhanggui-zhiku-agent"
    environment: str = "dev"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"

    database_url: str = "postgresql+asyncpg://zhiku:zhiku@localhost:5432/zhiku"
    redis_url: str = "redis://localhost:6379/0"

    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "zhiku_chunks"
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "zhiku_chunks"

    openai_base_url: str = "https://api.deepseek.com/v1"
    openai_api_key: str = "replace-me"
    chat_model: str = "deepseek-chat"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024
    rerank_url: str = ""
    rerank_top_n: int = 20
    internal_api_base_url: str = "http://localhost:8081"
    internal_api_token: str = ""
    tool_timeout_seconds: float = 3.0

    jwt_secret: str = "replace-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 720

    default_top_k: int = 8
    bm25_top_k: int = 40
    vector_top_k: int = 40
    rrf_k: int = 60
    max_agent_steps: int = 8

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
