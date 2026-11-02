# ============================================================
# RETRIEVAL SYSTEM
# ============================================================

import re

import cohere
from rank_bm25 import BM25Okapi

from config import (
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
    SEMANTIC_TOP_K,
    BM25_TOP_K,
    RRF_TOP_K,
    RRF_K,
    RERANK_TOP_K,
)


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Lowercase the text and split it into word tokens.

    The SAME tokenizer is used for indexing and for queries, so
    "Methodology," in a document matches "methodology" in a query.
    """

    return re.findall(r"\w+", text.lower())


# ============================================================
# CREATE BM25 SEARCH INDEX
# ============================================================

def create_bm25_index(chunks):
    """
    Create a BM25 keyword search index from document chunks.
    """

    corpus = [tokenize(chunk.page_content) for chunk in chunks]

    return BM25Okapi(corpus)


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(query, embeddings, collection):
    """
    Perform semantic/vector search using ChromaDB.

    Returns chunk IDs ordered from most to least similar.
    """

    total = collection.count()

    if total == 0:
        return []

    query_vector = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(SEMANTIC_TOP_K, total),
    )

    return results["ids"][0]


# ============================================================
# BM25 KEYWORD SEARCH
# ============================================================

def keyword_search(query, bm25, chunks):
    """
    Perform keyword-based search using BM25.

    Returns chunk IDs (as strings, matching the vector store IDs)
    ordered from best to worst. Chunks with a zero score share no
    terms with the query, so they are dropped instead of being
    returned as fake matches.
    """

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:BM25_TOP_K]

    return [str(i) for i in ranked if scores[i] > 0]


# ============================================================
# RECIPROCAL RANK FUSION (RRF)
# ============================================================

def reciprocal_rank_fusion(rankings, k=RRF_K):
    """
    Fuse several ranked lists of IDs into one ranking.

    Each item scores 1 / (k + rank) in every list it appears in;
    scores are summed. Returns [(id, score), ...] best first.
    """

    scores = {}

    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0) + 1 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def hybrid_search(semantic_ids, bm25_ids, chunks):
    """
    Combine semantic and BM25 results using RRF and return the
    top RRF_TOP_K candidate chunks (these go to the reranker).
    """

    fused = reciprocal_rank_fusion([semantic_ids, bm25_ids])

    return [chunks[int(chunk_id)] for chunk_id, _ in fused[:RRF_TOP_K]]


# ============================================================
# COHERE RERANKING
# ============================================================

_COHERE_CLIENT = None


def _get_cohere_client():
    global _COHERE_CLIENT

    if _COHERE_CLIENT is None:
        _COHERE_CLIENT = cohere.ClientV2(api_key=COHERE_API_KEY)

    return _COHERE_CLIENT


def rerank_results(query, candidate_chunks):
    """
    Rerank candidate chunks with Cohere and keep the best RERANK_TOP_K.
    """

    documents = [chunk.page_content for chunk in candidate_chunks]

    return _get_cohere_client().rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=min(RERANK_TOP_K, len(documents)),
    )
