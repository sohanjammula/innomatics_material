"""
Configuration file for Equipment Troubleshooting RAG (hybrid search)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Qdrant Cloud Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "https://your-cluster-id.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "your-api-key-here")
QDRANT_COLLECTION_NAME = "equipment_troubleshooting_manuals"

# Dense (semantic) embeddings

EMBEDDING_MODEL_TYPE="HUGGING_FACE"
DENSE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
DENSE_VECTOR_NAME = "dense"

# Sparse (keyword/BM25) embeddings - makes the search "hybrid"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
SPARSE_VECTOR_NAME = "sparse"

# LLM Configuration
LLM_MODEL = "openai/gpt-oss-120b"
LLM_PROVIDER = "openai"
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_API_KEY= os.getenv("GROQ_API_KEY")


# Document Processing Configuration
MANUALS_DIR = "Equipment Manuals"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Configuration
RETRIEVAL_K = 4  # Number of chunks to retrieve (spans 4 manuals)

# Chat History Configuration
REDIS_URL=os.getenv("REDIS_URL")
REDIS_PASSWORD=os.getenv("REDIS_PASSWORD")

REDIS_KEY_PREFIX = "equipment_chat"