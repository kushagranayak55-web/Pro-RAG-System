# ============================================================
# MAIN APPLICATION
# ============================================================

from loader import load_documents, create_chunks

from embeddings import (
    create_embedding_model,
    create_collection,
    store_embeddings
)

from retriever import (
    create_bm25_index,
    semantic_search,
    keyword_search,
    hybrid_search,
    rerank_results
)

from rag import (
    create_llm,
    rewrite_query,
    prepare_context,
    generate_answer
)


# ============================================================
# 1. LOAD AND CHUNK DOCUMENTS
# ============================================================

# PDF documents load kar rahe hain
documents = load_documents()

print("Total pages loaded:", len(documents))

# Documents ko smaller chunks mein divide kar rahe hain
chunks = create_chunks(documents)

print("Total chunks:", len(chunks))


# ============================================================
# 2. INITIALIZE EMBEDDINGS AND CHROMADB
# ============================================================

# Mistral embedding model create kar rahe hain
embeddings = create_embedding_model()

# ChromaDB collection create/load kar rahe hain
collection = create_collection()


# ============================================================
# 3. STORE CHUNKS IN CHROMADB
# ============================================================

# Chunks ke embeddings ChromaDB mein store kar rahe hain
count = store_embeddings(
    chunks,
    embeddings,
    collection
)

print("Chunks stored in ChromaDB:", count)


# ============================================================
# 4. CREATE BM25 INDEX
# ============================================================

# BM25 keyword search index create kar rahe hain
bm25 = create_bm25_index(chunks)


# ============================================================
# 5. INITIALIZE GROQ LLM
# ============================================================

# Groq LLM create kar rahe hain
llm = create_llm()


# ============================================================
# 6. START CHATBOT
# ============================================================

print("\n========================================")
print("        ADVANCED RAG CHATBOT")
print("========================================")
print("Type 'exit' or 'quit' to stop the chatbot.")


# Continuous questions ke liye loop
while True:

    # ========================================================
    # 7. GET USER QUERY
    # ========================================================

    original_query = input(
        "\nAsk your question: "
    ).strip()

    # Empty input ignore kar rahe hain
    if not original_query:
        continue

    # Exit command check kar rahe hain
    if original_query.lower() in [
        "exit",
        "quit"
    ]:

        print("\nChatbot stopped.")
        break


    # ========================================================
    # 8. QUERY REWRITING
    # ========================================================

    # User query ko retrieval-friendly query mein convert kar rahe hain
    rewritten_query = rewrite_query(
        original_query,
        llm
    )

    print(
        "\nOriginal Query:",
        original_query
    )

    print(
        "Rewritten Query:",
        rewritten_query
    )


    # ========================================================
    # 9. SEMANTIC SEARCH
    # ========================================================

    # ChromaDB se semantic results retrieve kar rahe hain
    semantic_ids = semantic_search(
        rewritten_query,
        embeddings,
        collection
    )


    # ========================================================
    # 10. KEYWORD SEARCH
    # ========================================================

    # BM25 se keyword-based results retrieve kar rahe hain
    bm25_results = keyword_search(
        rewritten_query,
        bm25,
        chunks
    )


    # ========================================================
    # 11. HYBRID SEARCH USING RRF
    # ========================================================

    # Semantic + BM25 results ko RRF se combine kar rahe hain
    candidate_chunks = hybrid_search(
        semantic_ids,
        bm25_results,
        chunks
    )


    # ========================================================
    # 12. COHERE RERANKING
    # ========================================================

    # Hybrid results ko Cohere relevance ke basis par rerank kar rahe hain
    rerank_response = rerank_results(
        rewritten_query,
        candidate_chunks
    )


    # ========================================================
    # 13. PREPARE CONTEXT
    # ========================================================

    # Top reranked chunks ko LLM context mein convert kar rahe hain
    context = prepare_context(
        rerank_response,
        candidate_chunks
    )


    # ========================================================
    # 14. GENERATE FINAL ANSWER
    # ========================================================

    # Retrieved context ke basis par final answer generate kar rahe hain
    answer = generate_answer(
        original_query,
        context,
        llm
    )


    # ========================================================
    # 15. DISPLAY FINAL ANSWER
    # ========================================================

    print("\n===== FINAL ANSWER =====")
    print(answer)