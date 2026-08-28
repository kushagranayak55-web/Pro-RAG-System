# ============================================================
# CONFIGURATION
# ============================================================

import os
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


# API Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Model Configuration
MISTRAL_EMBEDDING_MODEL = "mistral-embed"

GROQ_MODEL = "openai/gpt-oss-120b"

COHERE_RERANK_MODEL = "rerank-v3.5"


# RAG Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

SEMANTIC_TOP_K = 5
BM25_TOP_K = 5
RERANK_TOP_K = 3


# ChromaDB Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "advanced_rag"