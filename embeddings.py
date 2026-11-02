# ============================================================
# EMBEDDINGS AND VECTOR DATABASE
# ============================================================

from langchain_mistralai import MistralAIEmbeddings
import chromadb

from config import (
    MISTRAL_EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    SESSION_COLLECTION_PREFIX,
)


# ============================================================
# CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model():
    """
    Create and return the Mistral embedding model.
    """

    return MistralAIEmbeddings(
        model=MISTRAL_EMBEDDING_MODEL
    )


# ============================================================
# CHROMADB CLIENT / COLLECTIONS
# ============================================================

_CLIENT = None


def get_client():
    """
    Return one shared persistent ChromaDB client.
    """

    global _CLIENT

    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    return _CLIENT


def create_collection(collection_name=None):
    """
    Create or load a ChromaDB collection.
    """

    if collection_name is None:
        collection_name = COLLECTION_NAME

    return get_client().get_or_create_collection(name=collection_name)


def delete_collection(collection_name):
    """
    Delete a collection. Safe to call if it no longer exists.
    """

    try:
        get_client().delete_collection(name=collection_name)
    except Exception:
        # Already deleted or never created - nothing to clean up.
        pass


def cleanup_session_collections():
    """
    Delete every API-created collection (prefix 'rag_').

    Sessions live in memory, so after a server restart any
    'rag_*' collection on disk is orphaned and can be removed.
    Collections used by the CLI are not touched.
    """

    removed = 0

    for collection in get_client().list_collections():
        # Depending on the chromadb version this is an object or a string.
        name = getattr(collection, "name", collection)

        if str(name).startswith(SESSION_COLLECTION_PREFIX):
            delete_collection(str(name))
            removed += 1

    return removed


# ============================================================
# STORE DOCUMENT EMBEDDINGS
# ============================================================

def store_embeddings(chunks, embeddings, collection, batch_size=EMBED_BATCH_SIZE):
    """
    Embed all chunks in batches and store them in ChromaDB.

    Uses one embedding API call and one DB write per batch
    (instead of one per chunk). Upsert makes re-runs idempotent.
    """

    for start in range(0, len(chunks), batch_size):

        batch = chunks[start:start + batch_size]
        texts = [chunk.page_content for chunk in batch]

        vectors = embeddings.embed_documents(texts)

        collection.upsert(
            ids=[str(start + i) for i in range(len(batch))],
            embeddings=vectors,
            documents=texts,
            metadatas=[chunk.metadata for chunk in batch],
        )

    return collection.count()
