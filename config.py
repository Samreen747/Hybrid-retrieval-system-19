import os
from pydantic import BaseModel

class RAGConfig(BaseModel):
    # Free local embedding model (Runs locally on CPU/GPU)
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    
    # Free local LLM via Ollama
    LLM_MODEL: str = "qwen2.5:7b"  # or "llama3.2"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Cross-encoder reranker (Already free & local)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking Strategy
    PARENT_CHUNK_SIZE: int = 2000
    PARENT_CHUNK_OVERLAP: int = 200
    CHILD_CHUNK_SIZE: int = 400
    CHILD_CHUNK_OVERLAP: int = 50

    # Retrieval
    TOP_K_RETRIEVAL: int = 15
    TOP_K_RERANK: int = 3

    # Storage Paths
    CHROMA_PERSIST_DIR: str = "./storage/chroma_db"
    DOCSTORE_PATH: str = "./storage/docstore"

config = RAGConfig()