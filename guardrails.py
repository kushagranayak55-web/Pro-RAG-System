# ============================================================
# GUARDRAILS
# ============================================================
# Lightweight, rule-based protections (no extra dependencies).
# They are a first line of defence, NOT a guarantee:
#   1. Uploaded documents are scanned for prompt-injection phrases.
#   2. Questions are length-limited and cleaned.
#   3. LLM answers are checked for prompt leaks and for markdown images / HTML
#      (a known way to leak data out of a chat UI).

import re
import unicodedata

from config import MAX_QUESTION_CHARS


# ============================================================
# 1. PROMPT-INJECTION SCANNER (for text inside uploaded PDFs)
# ============================================================

_PATTERNS = {
    "override_instructions": r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|every\s+|the\s+|your\s+|my\s+|these\s+|those\s+)*(?:previous|prior|above|earlier|preceding|former|system|original)\s+(?:instructions?|prompts?|rules?|guidelines?|directions?|messages?|context)\b",
    "override_all": r"\b(?:ignore|disregard)\s+(?:all|any|every)\s+(?:instructions?|rules?|guidelines?|restrictions?)\b",
    "do_not_follow": r"\bdo\s+not\s+(?:follow|obey|listen\s+to)\s+(?:the\s+|your\s+|any\s+)*(?:previous|above|prior|system|original)\b",
    "role_change": r"\byou\s+are\s+now\s+(?:a|an|the|in|my|dan|free|unrestricted|jailbroken|no\s+longer)\b",
    "unrestricted_ai": r"\b(?:ai|assistant|chatbot)\s+(?:with)?out\s+(?:any\s+)?(?:rules|restrictions|limits|limitations|filters|guidelines|censorship)\b",
    "act_unrestricted": r"\b(?:act|behave|respond|pretend)\s+(?:as|like|that\s+you\s+are|you\s+are|to\s+be)\s+(?:if\s+)?[^.\n]{0,40}\b(?:without|no)\s+(?:any\s+)?(?:restrictions?|rules?|limits?|filters?|guidelines?)\b",
    "new_instructions": r"\b(?:new|updated|revised|real|actual)\s+(?:system\s+)?(?:instructions?|prompt)\s*:",
    "prompt_leak_request": r"\b(?:reveal|show|print|repeat|output|display|leak|disclose|tell\s+me)\s+(?:me\s+)?(?:your|the)\s+(?:full\s+|entire\s+|hidden\s+|initial\s+|original\s+)?(?:system\s+)?(?:prompt|instructions)\b",
    "repeat_above": r"\brepeat\s+(?:the\s+)?(?:text|words|everything|content|conversation)\s+(?:above|before)\b",
    "delimiter_spoof": r"</?\s*(?:context|document|documents|system|instructions?|assistant)\s*>",
}

_COMPILED = {name: re.compile(pattern) for name, pattern in _PATTERNS.items()}

# Invisible characters that can be used to hide or split a malicious phrase
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)


def normalize(text):
    """Lowercase, unify unicode look-alikes, drop invisible characters, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).lower()
    return re.sub(r"\s+", " ", text)


def find_injection(text):
    """Return the names of all injection patterns found in `text` (empty list = looks clean)."""
    clean = normalize(text)
    return [name for name, pattern in _COMPILED.items() if pattern.search(clean)]


def scan_chunks(chunks):
    """
    Scan document chunks. Returns {chunk_index: [pattern names]} for flagged chunks only.
    """
    flagged = {}

    for index, chunk in enumerate(chunks):
        hits = find_injection(chunk.page_content)
        if hits:
            flagged[index] = hits

    return flagged


def defang_delimiters(text):
    """Stop retrieved text from closing or opening our <context> block."""
    return re.sub(r"</?\s*context\s*>", "[tag removed]", text, flags=re.IGNORECASE)


# ============================================================
# 2. QUESTION CHECK
# ============================================================

def check_question(question):
    """
    Returns (clean_question, error_message). error_message is None when the question is fine.
    """
    # remove control characters (keep normal whitespace)
    question = "".join(ch for ch in question if ch in "\n\t " or unicodedata.category(ch)[0] != "C").strip()

    if not question:
        return question, "Question cannot be empty."

    if len(question) > MAX_QUESTION_CHARS:
        return question, f"Question is too long (maximum {MAX_QUESTION_CHARS} characters)."

    return question, None


# ============================================================
# 3. ANSWER CHECK
# ============================================================

LEAK_MESSAGE = "I can't share my internal instructions, but I'm happy to answer questions about your document."

# Distinctive phrases from our own prompts. If an answer contains them, the prompt leaked.
_PROMPT_FINGERPRINTS = [
    "untrusted document text",
    "security rules",
    "these instructions always take priority",
    "using only the information inside the <context>",
    "answer the user's question using only the provided context",
]

_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)|!\[[^\]]*\]\[[^\]]*\]")
_DANGEROUS_HTML = re.compile(r"<\s*/?\s*(?:img|script|iframe|object|embed|style|link|svg|form|input)\b[^>]*>", re.IGNORECASE)


def sanitize_answer(answer):
    """
    Returns (safe_answer, issues). `issues` lists what had to be changed.
    """
    issues = []
    lowered = answer.lower().replace("’", "'")

    if any(fingerprint in lowered for fingerprint in _PROMPT_FINGERPRINTS):
        return LEAK_MESSAGE, ["prompt_leak"]

    if _MARKDOWN_IMAGE.search(answer):
        answer = _MARKDOWN_IMAGE.sub("[image removed]", answer)
        issues.append("markdown_image_removed")

    if _DANGEROUS_HTML.search(answer):
        answer = _DANGEROUS_HTML.sub("", answer)
        issues.append("html_removed")

    return answer, issues
