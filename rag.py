# ============================================================
# RAG GENERATION PIPELINE
# ============================================================

from langchain_groq import ChatGroq

from config import GROQ_MODEL


# ============================================================
# CREATE LLM
# ============================================================

def create_llm():
    """
    Create and return the Groq LLM.
    """

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0
    )

    return llm


# ============================================================
# QUERY REWRITING
# ============================================================

def rewrite_query(
    original_query,
    llm
):
    """
    Rewrite the user's question into a
    retrieval-friendly search query.
    """

    # Query rewriting ke liye prompt
    rewrite_prompt = f"""
Rewrite the following user question into a clear and
information-rich search query for retrieving relevant
information from a research paper.

Keep the original meaning.
Return only the rewritten search query.

User question:
{original_query}
"""

    # Groq se rewritten query generate kar rahe hain
    rewritten_query = llm.invoke(
        rewrite_prompt
    ).content.strip()

    return rewritten_query


# ============================================================
# PREPARE CONTEXT
# ============================================================

def prepare_context(
    rerank_response,
    candidate_chunks
):
    """
    Convert reranked chunks into a single context
    that can be passed to the LLM.
    """

    # Top reranked chunks ko store karenge
    context_parts = []

    # Har reranked result process kar rahe hain
    for result in rerank_response.results:

        # Reranked result se original chunk retrieve kar rahe hain
        chunk = candidate_chunks[result.index]

        # Chunk text context mein add kar rahe hain
        context_parts.append(
            chunk.page_content
        )

    # Saare chunks ko ek single context mein combine kar rahe hain
    context = "\n\n".join(context_parts)

    return context


# ============================================================
# GENERATE FINAL ANSWER
# ============================================================

def generate_answer(
    original_query,
    context,
    llm
):
    """
    Generate the final answer using the retrieved context.
    """

    # User question + retrieved context ko LLM ko de rahe hain
    answer_prompt = f"""
You are a helpful research assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- Do not make up information.
- Give one clear and concise answer.
- If the context does not contain enough information, say:
  "I could not find the answer in the provided documents."

User Question:
{original_query}

Context:
{context}
"""

    # Final answer generate kar rahe hain
    response = llm.invoke(
        answer_prompt
    )

    return response.content