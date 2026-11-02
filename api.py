# ============================================================
# ProRAG - FASTAPI BACKEND
# ============================================================
# Exposes the RAG pipeline (loader.py, embeddings.py, retriever.py,
# rag.py) as HTTP endpoints for the React frontend.
#
# Run with a single worker:  uvicorn api:app --port 8000
# (sessions are kept in this process's memory)

import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    ALLOWED_ORIGINS,
    MAX_UPLOAD_MB,
    SESSION_COLLECTION_PREFIX,
    SESSION_TTL_SECONDS,
)

from loader import load_documents, create_chunks

from embeddings import (
    create_embedding_model,
    create_collection,
    delete_collection,
    cleanup_session_collections,
    store_embeddings,
)

from retriever import (
    create_bm25_index,
    semantic_search,
    keyword_search,
    hybrid_search,
    rerank_results,
)

from rag import (
    NOT_FOUND_MESSAGE,
    create_llm,
    rewrite_query,
    prepare_context,
    generate_answer,
)


logger = logging.getLogger("prorag")


# ============================================================
# APP SETUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sessions are in memory, so any 'rag_*' collection left on disk
    # from a previous run is orphaned. Remove it on startup.
    removed = cleanup_session_collections()
    if removed:
        logger.info("Removed %d orphaned session collection(s).", removed)
    yield


app = FastAPI(title="ProRAG API", lifespan=lifespan)

# Only the configured frontend origins may call this API.
# Set ALLOWED_ORIGINS (comma-separated) to your deployed frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


# ============================================================
# MODELS ARE CREATED ONCE
# ============================================================

EMBEDDINGS = None
LLM = None


def get_embeddings():
    global EMBEDDINGS
    if EMBEDDINGS is None:
        EMBEDDINGS = create_embedding_model()
    return EMBEDDINGS


def get_llm():
    global LLM
    if LLM is None:
        LLM = create_llm()
    return LLM


# ============================================================
# IN-MEMORY SESSION STORE
# ============================================================
# Each uploaded PDF gets its own isolated session: its own temp
# directory, its own ChromaDB collection, its own chunks and
# BM25 index. Sessions never share retrieval context, and idle
# sessions are deleted after SESSION_TTL_SECONDS.

SESSIONS = {}
SESSIONS_LOCK = threading.Lock()


def _destroy_session(session):
    """Remove everything a session owns: temp files and its vector collection."""
    shutil.rmtree(session["pdf_directory"], ignore_errors=True)
    delete_collection(session["collection_name"])


def _purge_expired_sessions():
    """Delete sessions that have been idle longer than the TTL."""
    cutoff = time.time() - SESSION_TTL_SECONDS

    with SESSIONS_LOCK:
        expired_ids = [
            sid for sid, s in SESSIONS.items() if s["last_used"] < cutoff
        ]
        expired = [SESSIONS.pop(sid) for sid in expired_ids]

    for session in expired:
        _destroy_session(session)


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class AskRequest(BaseModel):
    session_id: str
    question: str


class ChunkPreview(BaseModel):
    number: int                      # matches the [n] citation in the answer
    text: str
    source: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None


class AskResponse(BaseModel):
    session_id: str
    original_query: str
    rewritten_query: str
    answer: str
    sources: list[dict]
    pipeline: dict
    reranked_chunks: list[ChunkPreview]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ============================================================
# UPLOAD A PDF -> RUN INGESTION PIPELINE -> CREATE SESSION
# ============================================================
# Declared as a plain `def` (not `async def`) on purpose: embedding
# and indexing are blocking calls, so FastAPI runs this endpoint in
# its threadpool instead of freezing the event loop.

@app.post("/api/upload")
def upload_pdf(file: UploadFile = File(...)):

    _purge_expired_sessions()

    # ----------------------------------------------------------
    # Validate the upload (name, type, size)
    # ----------------------------------------------------------
    # basename() prevents path traversal via filenames like "../../x.pdf"
    safe_name = os.path.basename(file.filename or "")

    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    contents = file.file.read(max_bytes + 1)

    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_MB} MB.",
        )

    if not contents.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="This file is not a valid PDF.")

    # ----------------------------------------------------------
    # Save the upload into its own isolated temp directory
    # ----------------------------------------------------------
    pdf_directory = tempfile.mkdtemp(prefix="prorag_")

    with open(os.path.join(pdf_directory, safe_name), "wb") as f:
        f.write(contents)

    file_size_mb = round(len(contents) / (1024 * 1024), 2)
    collection_name = None

    try:
        # ------------------------------------------------------
        # LOAD + CHUNK
        # ------------------------------------------------------
        documents = load_documents(pdf_directory)

        # A scanned PDF yields one empty Document per page, so check the text itself.
        if not any(doc.page_content.strip() for doc in documents):
            raise HTTPException(
                status_code=422,
                detail="Could not extract any text from this PDF (it may be a scanned image).",
            )

        chunks = create_chunks(documents)

        if not chunks:
            raise HTTPException(status_code=422, detail="The PDF produced no usable text chunks.")

        # ------------------------------------------------------
        # EMBED + STORE IN A FRESH, ISOLATED CHROMA COLLECTION
        # ------------------------------------------------------
        collection_name = SESSION_COLLECTION_PREFIX + uuid.uuid4().hex[:16]
        collection = create_collection(collection_name)

        stored_count = store_embeddings(chunks, get_embeddings(), collection)

        # ------------------------------------------------------
        # BM25 INDEX
        # ------------------------------------------------------
        bm25 = create_bm25_index(chunks)

    except Exception as e:
        # Undo everything created so far, then report the error.
        shutil.rmtree(pdf_directory, ignore_errors=True)
        if collection_name:
            delete_collection(collection_name)

        if isinstance(e, HTTPException):
            raise

        logger.exception("Failed to process document")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    session_id = uuid.uuid4().hex

    with SESSIONS_LOCK:
        SESSIONS[session_id] = {
            "pdf_directory": pdf_directory,
            "collection_name": collection_name,
            "collection": collection,
            "chunks": chunks,
            "bm25": bm25,
            "filename": safe_name,
            "last_used": time.time(),
        }

    return {
        "session_id": session_id,
        "filename": safe_name,
        "file_size_mb": file_size_mb,
        "num_pages": len(documents),
        "num_chunks": len(chunks),
        "stored_count": stored_count,
    }


# ============================================================
# ASK A QUESTION -> RUN THE FULL RAG PIPELINE
# ============================================================

@app.post("/api/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):

    _purge_expired_sessions()

    with SESSIONS_LOCK:
        session = SESSIONS.get(payload.session_id)
        if session is not None:
            session["last_used"] = time.time()

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Please upload a PDF again.",
        )

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    llm = get_llm()
    embeddings = get_embeddings()

    try:
        # 1. QUERY REWRITING
        rewritten_query = rewrite_query(question, llm)

        # 2. SEMANTIC SEARCH
        semantic_ids = semantic_search(rewritten_query, embeddings, session["collection"])

        # 3. BM25 KEYWORD SEARCH
        bm25_ids = keyword_search(rewritten_query, session["bm25"], session["chunks"])

        # 4. RRF HYBRID FUSION
        candidate_chunks = hybrid_search(semantic_ids, bm25_ids, session["chunks"])

        if candidate_chunks:
            # 5. COHERE RERANKING
            rerank_response = rerank_results(rewritten_query, candidate_chunks)

            # 6. CONTEXT PREPARATION (numbered, with source + page)
            context, sources = prepare_context(rerank_response, candidate_chunks)

            # 7. GROQ GENERATION
            answer = generate_answer(question, context, llm)
        else:
            rerank_response = None
            sources = []
            answer = NOT_FOUND_MESSAGE

    except Exception as e:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=500,
            detail=f"Something went wrong while generating your answer: {e}",
        )

    # Every count and score below comes directly from the pipeline run above.
    reranked_chunks = []

    if rerank_response is not None:
        for number, result in enumerate(rerank_response.results, start=1):
            chunk = candidate_chunks[result.index]
            reranked_chunks.append(
                ChunkPreview(
                    number=number,
                    text=chunk.page_content,
                    source=chunk.metadata.get("source"),
                    page=chunk.metadata.get("page"),
                    score=round(float(result.relevance_score), 4),
                )
            )

    pipeline = {
        "query_rewriting": {"status": "complete"},
        "semantic_search": {"status": "complete", "matches": len(semantic_ids)},
        "bm25_search": {"status": "complete", "matches": len(bm25_ids)},
        "rrf_fusion": {"status": "complete", "candidates": len(candidate_chunks)},
        "reranking": {"status": "complete", "selected": len(reranked_chunks)},
        "generation": {"status": "complete"},
    }

    return AskResponse(
        session_id=payload.session_id,
        original_query=question,
        rewritten_query=rewritten_query,
        answer=answer,
        sources=sources,
        pipeline=pipeline,
        reranked_chunks=reranked_chunks,
    )


# ============================================================
# DELETE A SESSION (used when the user uploads a new PDF)
# ============================================================

@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):

    with SESSIONS_LOCK:
        session = SESSIONS.pop(session_id, None)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Removes the temp PDF directory AND the ChromaDB collection.
    _destroy_session(session)

    return {"deleted": session_id}
