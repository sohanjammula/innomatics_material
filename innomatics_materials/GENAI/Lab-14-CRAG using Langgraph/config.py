"""
Configuration file for Corrective RAG using Langgraph
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant Cloud Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "solarnova_customer_support"

# OpenAI Embeddings Configuration
EMBEDDING_MODEL_TYPE="HUGGING_FACE"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# LLM Configuration
LLM_PROVIDER = "groq"  # default: Groq
# LLM_PROVIDER = "openai"  # fallback

# Main RAG LLM
RAG_LLM_MODEL = "llama-3.3-70b-versatile"  # default: Groq
# RAG_LLM_MODEL = "gpt-4o"  # fallback (use with LLM_PROVIDER = "openai")
RAG_LLM_TEMPERATURE = 0.1

# Question Rewriter LLM
REWRITER_LLM_MODEL = "llama-3.1-8b-instant"  # default: Groq
# REWRITER_LLM_MODEL = "gpt-4o-mini"  # fallback (use with LLM_PROVIDER = "openai")
REWRITER_LLM_TEMPERATURE = 0.1

# Retrieval Grader LLM -- Validation LLM
GRADER_LLM_MODEL = "gpt-5-mini"  # fallback (use with LLM_PROVIDER = "openai")
GRADER_LLM_TEMPERATURE = 0
GRADER_LLM_PROVIDER = "openai"

# Retrieval Configuration
RETRIEVAL_K = 3  # Number of chunks to retrieve


