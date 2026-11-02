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

# Stage 1: how many candidates each retriever returns.
# Retrieval is optimised for RECALL, so these are deliberately generous.
SEMANTIC_TOP_K = 20
BM25_TOP_K = 20

# Stage 2: how many fused (RRF) candidates are sent to the reranker.
RRF_TOP_K = 15

# Stage 3: how many reranked chunks become the final LLM context.
RERANK_TOP_K = 5

# RRF smoothing constant (60 is the value from the original RRF paper)
RRF_K = 60

# Embedding batch size (chunks per embedding API call)
EMBED_BATCH_SIZE = 32


# ChromaDB Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "advanced_rag"

# Collections created by the API are prefixed so they can be
# identified and cleaned up safely (the CLI uses COLLECTION_NAME).
SESSION_COLLECTION_PREFIX = "rag_"


# API / Security Configuration
# Comma-separated list of allowed frontend origins (CORS).
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

# Maximum accepted upload size in megabytes.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))

# Sessions idle for longer than this are deleted automatically.
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
