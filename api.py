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
from langsmith import traceable
from pydantic import BaseModel

from config import (
    ALLOWED_ORIGINS,
    DROP_FLAGGED_CHUNKS,
    GUARDRAILS_ENABLED,
    MAX_UPLOAD_MB,
    SESSION_COLLECTION_PREFIX,
    SESSION_TTL_SECONDS,
)

from guardrails import scan_chunks, check_question, sanitize_answer

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
    guardrails: dict = {}


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

        # ------------------------------------------------------
        # GUARDRAIL: scan the document for prompt-injection text
        # ------------------------------------------------------
        flagged = scan_chunks(chunks) if GUARDRAILS_ENABLED else {}

        for index in flagged:
            chunks[index].metadata["flagged"] = True

        if flagged:
            logger.warning(
                "Possible prompt injection in %s: %d chunk(s) flagged %s",
                safe_name, len(flagged), sorted({n for names in flagged.values() for n in names}),
            )

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
        "flagged_chunks": len(flagged),
    }


# ============================================================
# TRACED PIPELINE STEPS (LangSmith)
# ============================================================
# Each step is a small function with light inputs (strings / ids), so a LangSmith
# trace shows one clear span per stage. Tracing is OFF unless you set
#   LANGSMITH_TRACING=true  and  LANGSMITH_API_KEY=...
# Note: when tracing is on, questions and document text are sent to LangSmith.

def _session(session_id):
    with SESSIONS_LOCK:
        return SESSIONS[session_id]


@traceable(name="query_rewriting", run_type="chain")
def _step_rewrite(question: str):
    return rewrite_query(question, get_llm())


@traceable(name="semantic_search", run_type="retriever")
def _step_semantic(query: str, session_id: str):
    return semantic_search(query, get_embeddings(), _session(session_id)["collection"])


@traceable(name="bm25_search", run_type="retriever")
def _step_bm25(query: str, session_id: str):
    session = _session(session_id)
    return keyword_search(query, session["bm25"], session["chunks"])


@traceable(name="rrf_fusion", run_type="chain")
def _step_fuse(semantic_ids: list, bm25_ids: list, session_id: str):
    return hybrid_search(semantic_ids, bm25_ids, _session(session_id)["chunks"])


@traceable(name="guardrail_filter", run_type="chain")
def _step_filter_flagged(candidates: list):
    """Never send chunks that look like prompt injection to the reranker or the LLM."""
    if not (GUARDRAILS_ENABLED and DROP_FLAGGED_CHUNKS):
        return candidates, 0

    kept = [c for c in candidates if not c.metadata.get("flagged")]

    return kept, len(candidates) - len(kept)


@traceable(name="cohere_rerank", run_type="retriever")
def _step_rerank(query: str, candidates: list):
    return rerank_results(query, candidates)


@traceable(name="generation", run_type="chain")
def _step_generate(question: str, rerank_response, candidates: list):
    context, sources = prepare_context(rerank_response, candidates)
    return generate_answer(question, context, get_llm()), sources


@traceable(name="prorag_pipeline", run_type="chain")
def run_rag_pipeline(question: str, session_id: str):
    rewritten_query = _step_rewrite(question)
    semantic_ids = _step_semantic(rewritten_query, session_id)
    bm25_ids = _step_bm25(rewritten_query, session_id)

    fused = _step_fuse(semantic_ids, bm25_ids, session_id)
    candidates, dropped = _step_filter_flagged(fused)

    if candidates:
        rerank_response = _step_rerank(rewritten_query, candidates)
        answer, sources = _step_generate(question, rerank_response, candidates)
    else:
        rerank_response, sources, answer = None, [], NOT_FOUND_MESSAGE

    return {
        "rewritten_query": rewritten_query,
        "semantic_ids": semantic_ids,
        "bm25_ids": bm25_ids,
        "candidates": candidates,
        "dropped": dropped,
        "rerank_response": rerank_response,
        "sources": sources,
        "answer": answer,
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

    # GUARDRAIL: clean the question and enforce the length limit
    if GUARDRAILS_ENABLED:
        question, problem = check_question(payload.question)
    else:
        question, problem = payload.question.strip(), None
        problem = None if question else "Question cannot be empty."

    if problem:
        raise HTTPException(status_code=400, detail=problem)

    try:
        out = run_rag_pipeline(question, payload.session_id)
    except Exception as e:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=500,
            detail=f"Something went wrong while generating your answer: {e}",
        )

    # GUARDRAIL: check the answer (prompt leak, markdown images, dangerous HTML)
    answer, issues = (sanitize_answer(out["answer"]) if GUARDRAILS_ENABLED else (out["answer"], []))

    if issues:
        logger.warning("Answer modified by guardrails: %s", issues)

    candidate_chunks = out["candidates"]
    rerank_response = out["rerank_response"]

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
        "semantic_search": {"status": "complete", "matches": len(out["semantic_ids"])},
        "bm25_search": {"status": "complete", "matches": len(out["bm25_ids"])},
        "rrf_fusion": {"status": "complete", "candidates": len(candidate_chunks)},
        "reranking": {"status": "complete", "selected": len(reranked_chunks)},
        "generation": {"status": "complete"},
    }

    return AskResponse(
        session_id=payload.session_id,
        original_query=question,
        rewritten_query=out["rewritten_query"],
        answer=answer,
        sources=out["sources"],
        pipeline=pipeline,
        reranked_chunks=reranked_chunks,
        guardrails={
            "enabled": GUARDRAILS_ENABLED,
            "flagged_chunks_dropped": out["dropped"],
            "answer_issues": issues,
        },
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
