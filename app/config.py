"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Tier-1 Configuration (cheap/free model for classification)
    tier1_provider: Literal["groq", "google", "openai"] = "groq"
    tier1_model: str = "llama-3.1-8b-instant"
    
    # Tier-2 Configuration (paid model for synthesis)
    tier2_provider: Literal["openai", "anthropic"] = "openai"
    tier2_model: str = "gpt-4o"
    
    # API Keys
    groq_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Database Configuration
    chroma_persist_dir: Path = Path("./data/chroma")
    sqlite_db_path: Path = Path("./data/pharma.db")
    
    # Application Settings
    debug: bool = False
    log_level: str = "INFO"
    
    # Retrieval Settings
    vector_top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def get_tier1_api_key(self) -> str:
        """Get the API key for Tier-1 provider."""
        if self.tier1_provider == "groq":
            return self.groq_api_key
        elif self.tier1_provider == "google":
            return self.google_api_key
        elif self.tier1_provider == "openai":
            return self.openai_api_key
        return ""
    
    def get_tier2_api_key(self) -> str:
        """Get the API key for Tier-2 provider."""
        if self.tier2_provider == "openai":
            return self.openai_api_key
        elif self.tier2_provider == "anthropic":
            return self.anthropic_api_key
        return ""


# Global settings instance
settings = Settings()
