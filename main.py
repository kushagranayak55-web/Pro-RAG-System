# ============================================================
# COMMAND-LINE RAG CHATBOT
# ============================================================
# Indexes every PDF in the ./data folder and starts an
# interactive question-answering loop in the terminal.
#
#   python main.py
#
# (The web app uses api.py instead.)

import sys
from pathlib import Path

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
    NOT_FOUND_MESSAGE,
    create_llm,
    rewrite_query,
    prepare_context,
    generate_answer
)


def main():

    # --------------------------------------------------------
    # 1. LOAD AND CHUNK DOCUMENTS
    # --------------------------------------------------------

    if not Path("data").is_dir() or not list(Path("data").glob("*.pdf")):
        sys.exit("Put at least one PDF into the ./data folder and run again.")

    documents = load_documents()
    print("Total pages loaded:", len(documents))

    chunks = create_chunks(documents)
    print("Total chunks:", len(chunks))

    # --------------------------------------------------------
    # 2. EMBEDDINGS + CHROMADB
    # --------------------------------------------------------

    embeddings = create_embedding_model()
    collection = create_collection()

    # --------------------------------------------------------
    # 3. STORE CHUNKS (batched, safe to re-run)
    # --------------------------------------------------------

    count = store_embeddings(chunks, embeddings, collection)
    print("Chunks stored in ChromaDB:", count)

    # --------------------------------------------------------
    # 4. BM25 INDEX + LLM
    # --------------------------------------------------------

    bm25 = create_bm25_index(chunks)
    llm = create_llm()

    # --------------------------------------------------------
    # 5. CHAT LOOP
    # --------------------------------------------------------

    print("\n========================================")
    print("        ADVANCED RAG CHATBOT")
    print("========================================")
    print("Type 'exit' or 'quit' to stop the chatbot.")

    while True:

        original_query = input("\nAsk your question: ").strip()

        if not original_query:
            continue

        if original_query.lower() in ["exit", "quit"]:
            print("\nChatbot stopped.")
            break

        # Query rewriting
        rewritten_query = rewrite_query(original_query, llm)

        print("\nOriginal Query:", original_query)
        print("Rewritten Query:", rewritten_query)

        # Retrieval: semantic + BM25 -> RRF fusion
        semantic_ids = semantic_search(rewritten_query, embeddings, collection)
        bm25_ids = keyword_search(rewritten_query, bm25, chunks)
        candidate_chunks = hybrid_search(semantic_ids, bm25_ids, chunks)

        if not candidate_chunks:
            print("\n===== FINAL ANSWER =====")
            print(NOT_FOUND_MESSAGE)
            continue

        # Rerank -> context -> answer
        rerank_response = rerank_results(rewritten_query, candidate_chunks)
        context, sources = prepare_context(rerank_response, candidate_chunks)
        answer = generate_answer(original_query, context, llm)

        print("\n===== FINAL ANSWER =====")
        print(answer)

        print("\nSources:")
        for s in sources:
            print(f"  [{s['number']}] {s['source']}, page {s['page']}")


if __name__ == "__main__":
    main()
