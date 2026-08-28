# ============================================================
# RETRIEVAL SYSTEM
# ============================================================

import os

import cohere
from rank_bm25 import BM25Okapi

from config import (
    SEMANTIC_TOP_K,
    BM25_TOP_K,
    RERANK_TOP_K,
    COHERE_RERANK_MODEL
)


# ============================================================
# CREATE BM25 SEARCH INDEX
# ============================================================

def create_bm25_index(chunks):
    """
    Create a BM25 keyword search index from document chunks.
    """

    # Har chunk ko words/tokens mein convert kar rahe hain
    corpus = [
        chunk.page_content.split()
        for chunk in chunks
    ]

    # BM25 index create kar rahe hain
    bm25 = BM25Okapi(corpus)

    return bm25


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    embeddings,
    collection
):
    """
    Perform semantic/vector search using ChromaDB.
    """

    # Query ko embedding/vector mein convert kar rahe hain
    query_vector = embeddings.embed_query(query)

    # ChromaDB se semantically relevant chunks retrieve kar rahe hain
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=SEMANTIC_TOP_K
    )

    # Retrieved chunk IDs return kar rahe hain
    return results["ids"][0]


# ============================================================
# BM25 KEYWORD SEARCH
# ============================================================

def keyword_search(
    query,
    bm25,
    chunks
):
    """
    Perform keyword-based search using BM25.
    """

    # Query ko tokens mein divide kar rahe hain
    query_tokens = query.split()

    # BM25 se top relevant chunks retrieve kar rahe hain
    results = bm25.get_top_n(
        query_tokens,
        chunks,
        n=BM25_TOP_K
    )

    return results


# ============================================================
# RECIPROCAL RANK FUSION (RRF)
# ============================================================

def hybrid_search(
    semantic_ids,
    bm25_results,
    chunks
):
    """
    Combine semantic search and BM25 results
    using Reciprocal Rank Fusion.
    """

    # Har chunk ka combined RRF score store karenge
    rrf_scores = {}

    # Chunk object ko uske ID ke saath map kar rahe hain
    chunk_to_id = {
        id(chunk): str(i)
        for i, chunk in enumerate(chunks)
    }


    # --------------------------------------------------------
    # Semantic Search Ranking
    # --------------------------------------------------------

    for rank, chunk_id in enumerate(
        semantic_ids,
        start=1
    ):

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (60 + rank)
        )


    # --------------------------------------------------------
    # BM25 Ranking
    # --------------------------------------------------------

    for rank, chunk in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = chunk_to_id[id(chunk)]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (60 + rank)
        )


    # --------------------------------------------------------
    # Sort by RRF Score
    # --------------------------------------------------------

    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # Top hybrid candidates return kar rahe hain
    candidate_chunks = [
        chunks[int(chunk_id)]
        for chunk_id, score in ranked_results[:SEMANTIC_TOP_K]
    ]

    return candidate_chunks


# ============================================================
# COHERE RERANKING
# ============================================================

def rerank_results(
    query,
    candidate_chunks
):
    """
    Rerank retrieved chunks using Cohere.
    """

    # Cohere client create kar rahe hain
    co = cohere.ClientV2(
        api_key=os.getenv("COHERE_API_KEY")
    )

    # Chunks ka text Cohere ko provide kar rahe hain
    documents = [
        chunk.page_content
        for chunk in candidate_chunks
    ]

    # Cohere relevance ke basis par chunks ko rerank kar raha hai
    rerank_response = co.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=RERANK_TOP_K
    )

    return rerank_response