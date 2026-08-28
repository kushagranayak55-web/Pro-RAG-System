# ============================================================
# EMBEDDINGS AND VECTOR DATABASE
# ============================================================

from langchain_mistralai import MistralAIEmbeddings
import chromadb

from config import (
    MISTRAL_EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    COLLECTION_NAME
)


# ============================================================
# CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model():
    """
    Create and return the Mistral embedding model.
    """

    embeddings = MistralAIEmbeddings(
        model=MISTRAL_EMBEDDING_MODEL
    )

    return embeddings


# ============================================================
# CREATE CHROMADB COLLECTION
# ============================================================
# ============================================================
# CREATE CHROMADB COLLECTION
# ============================================================

def create_collection(collection_name=None):
    """
    Create or load a ChromaDB collection.
    """

    # Persistent ChromaDB client
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    # Agar custom collection name diya hai
    # to usi naam ka collection create/load hoga
    if collection_name is None:
        collection_name = COLLECTION_NAME

    collection = client.get_or_create_collection(
        name=collection_name
    )

    return collection


# ============================================================
# STORE DOCUMENT EMBEDDINGS
# ============================================================

def store_embeddings(chunks, embeddings, collection):
    """
    Generate embeddings for all chunks
    and store them in ChromaDB.
    """

    # Store every chunk in ChromaDB
    for i, chunk in enumerate(chunks):

        # Convert chunk text into vector
        vector = embeddings.embed_query(
            chunk.page_content
        )

        # Store vector, text and metadata
        collection.add(
            ids=[str(i)],
            embeddings=[vector],
            documents=[chunk.page_content],
            metadatas=[chunk.metadata]
        )

    return collection.count()