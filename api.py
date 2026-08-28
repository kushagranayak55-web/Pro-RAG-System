
# This file does NOT reimplement or modify any RAG logic.
# It only imports and calls the existing functions from
# loader.py, embeddings.py, retriever.py and rag.py exactly
# as the original Streamlit app did, and exposes them as
# HTTP endpoints so the React frontend can call them.

import os
import shutil
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from loader import load_documents, create_chunks

from embeddings import (
    create_embedding_model,
    create_collection,
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
    create_llm,
    rewrite_query,
    prepare_context,
    generate_answer,
)


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="ProRAG API")

# Allow the Vite dev server (and any origin during local dev) to call this API.
# Tighten allow_origins to your real frontend domain before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELS ARE CREATED ONCE (mirrors the old st.session_state caching)
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
# BM25 index. Sessions never share retrieval context.

SESSIONS = {}


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class AskRequest(BaseModel):
    session_id: str
    question: str


class ChunkPreview(BaseModel):
    text: str
    page: Optional[int] = None
    score: Optional[float] = None


class AskResponse(BaseModel):
    session_id: str
    original_query: str
    rewritten_query: str
    answer: str
    pipeline: dict
    reranked_chunks: list[ChunkPreview]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ============================================================
# UPLOAD A PDF → RUN INGESTION PIPELINE → CREATE SESSION
# ============================================================

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # ------------------------------------------------------
    # Save the upload into its own isolated temp directory
    # ------------------------------------------------------
    pdf_directory = tempfile.mkdtemp(prefix="prorag_")
    pdf_path = os.path.join(pdf_directory, file.filename)

    contents = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(contents)

    file_size_mb = round(len(contents) / (1024 * 1024), 2)

    try:
        # ----------------------------------------------------
        # LOAD + CHUNK
        # ----------------------------------------------------
        documents = load_documents(pdf_directory)

        if not documents:
            raise HTTPException(status_code=422, detail="Could not extract any text from this PDF.")

        chunks = create_chunks(documents)

        # ----------------------------------------------------
        # EMBED + STORE IN A FRESH, ISOLATED CHROMA COLLECTION
        # ----------------------------------------------------
        collection_name = "rag_" + uuid.uuid4().hex[:16]
        collection = create_collection(collection_name)

        stored_count = store_embeddings(chunks, get_embeddings(), collection)

        # ----------------------------------------------------
        # BM25 INDEX
        # ----------------------------------------------------
        bm25 = create_bm25_index(chunks)

    except HTTPException:
        shutil.rmtree(pdf_directory, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(pdf_directory, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    session_id = uuid.uuid4().hex

    SESSIONS[session_id] = {
        "pdf_directory": pdf_directory,
        "collection_name": collection_name,
        "collection": collection,
        "chunks": chunks,
        "bm25": bm25,
        "filename": file.filename,
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "file_size_mb": file_size_mb,
        "num_pages": len(documents),
        "num_chunks": len(chunks),
        "stored_count": stored_count,
    }


# ============================================================
# ASK A QUESTION → RUN THE FULL RAG PIPELINE
# ============================================================

@app.post("/api/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):

    session = SESSIONS.get(payload.session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")

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
        bm25_results = keyword_search(rewritten_query, session["bm25"], session["chunks"])

        # 4. RRF HYBRID FUSION
        candidate_chunks = hybrid_search(semantic_ids, bm25_results, session["chunks"])

        # 5. COHERE RERANKING
        rerank_response = rerank_results(rewritten_query, candidate_chunks)

        # 6. CONTEXT PREPARATION
        context = prepare_context(rerank_response, candidate_chunks)

        # 7. GROQ GENERATION
        answer = generate_answer(question, context, llm)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong while generating your answer: {e}")

    # Build a truthful record of what actually happened at each stage —
    # every count and score below comes directly from the pipeline run above.
    reranked_chunks = []
    for result in rerank_response.results:
        chunk = candidate_chunks[result.index]
        reranked_chunks.append(
            ChunkPreview(
                text=chunk.page_content,
                page=chunk.metadata.get("page"),
                score=round(float(result.relevance_score), 4),
            )
        )

    pipeline = {
        "query_rewriting": {"status": "complete"},
        "semantic_search": {"status": "complete", "matches": len(semantic_ids)},
        "bm25_search": {"status": "complete", "matches": len(bm25_results)},
        "rrf_fusion": {"status": "complete", "candidates": len(candidate_chunks)},
        "reranking": {"status": "complete", "selected": len(rerank_response.results)},
        "generation": {"status": "complete"},
    }

    return AskResponse(
        session_id=payload.session_id,
        original_query=question,
        rewritten_query=rewritten_query,
        answer=answer,
        pipeline=pipeline,
        reranked_chunks=reranked_chunks,
    )


# ============================================================
# DELETE A SESSION (used when the user uploads a new PDF)
# ============================================================

@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):

    session = SESSIONS.pop(session_id, None)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    shutil.rmtree(session["pdf_directory"], ignore_errors=True)

    return {"deleted": session_id}
