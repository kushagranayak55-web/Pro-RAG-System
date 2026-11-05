"""Tests for the retrieval code (tokenizer, BM25, RRF, batching, context building). No network needed."""
import math
import types

import chromadb
from langchain_core.documents import Document

from embeddings import store_embeddings
from rag import prepare_context
from retriever import (
    create_bm25_index,
    hybrid_search,
    keyword_search,
    reciprocal_rank_fusion,
    tokenize,
)


def make_chunks(texts):
    return [Document(page_content=t, metadata={"source": "doc.pdf", "page": i + 1}) for i, t in enumerate(texts)]


def test_tokenizer_lowercases_and_strips_punctuation():
    assert tokenize("Methodology, Equivariance-Error!") == ["methodology", "equivariance", "error"]


def test_bm25_matches_regardless_of_case_and_punctuation():
    chunks = make_chunks(["The Methodology, in short.", "Results were good.", "Unrelated text"])
    assert keyword_search("methodology", create_bm25_index(chunks), chunks) == ["0"]


def test_bm25_drops_chunks_that_share_no_words_with_the_query():
    chunks = make_chunks(["alpha beta", "gamma delta"])
    assert keyword_search("zzzz", create_bm25_index(chunks), chunks) == []
    assert keyword_search("", create_bm25_index(chunks), chunks) == []


def test_rrf_rewards_items_that_appear_in_both_lists():
    fused = [item for item, _ in reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])]
    assert fused[:2] == ["a", "c"]
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_scores_follow_the_formula():
    (item, score), = reciprocal_rank_fusion([["a"], ["a"]], k=60)
    assert item == "a" and math.isclose(score, 2 / 61)


def test_hybrid_search_maps_ids_back_to_chunks():
    chunks = make_chunks(["zero", "one", "two"])
    result = hybrid_search(["2", "0"], ["2"], chunks)
    assert [c.page_content for c in result][0] == "two"


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[float(len(t)), 1.0] for t in texts]


def test_embeddings_are_sent_in_batches_not_one_by_one():
    chunks = make_chunks([f"chunk number {i}" for i in range(70)])
    emb = FakeEmbeddings()
    collection = chromadb.EphemeralClient().get_or_create_collection("test_batching")
    count = store_embeddings(chunks, emb, collection, batch_size=32)
    assert count == 70
    assert emb.calls == 3          # 32 + 32 + 6, not 70 calls


def test_prepare_context_numbers_chunks_and_keeps_source_and_page():
    chunks = make_chunks(["first text", "second text"])
    response = types.SimpleNamespace(results=[types.SimpleNamespace(index=1), types.SimpleNamespace(index=0)])
    context, sources = prepare_context(response, chunks)
    assert context.startswith("[1] (doc.pdf, page 2)\nsecond text")
    assert sources == [
        {"number": 1, "source": "doc.pdf", "page": 2},
        {"number": 2, "source": "doc.pdf", "page": 1},
    ]
