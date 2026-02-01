import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval settings
    SIMILARITY_THRESHOLD: float = 0.75
    MAX_CONTEXT_LENGTH: int = 4000
    DIVERSITY_THRESHOLD: float = 0.9
    
    # Vector DB
    PERSIST_DIRECTORY: str = "./vector_db"
    
    # LLM settings
    USE_OPENAI: bool = False
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        import dotenv
        dotenv.load_dotenv()
        
        return cls(
            USE_OPENAI=os.getenv("USE_OPENAI", "false").lower() == "true"
        )

# Global config instance
config = Config.from_env()