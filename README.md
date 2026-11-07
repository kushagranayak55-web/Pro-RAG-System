# 🚀 Pro-RAG System

> **Advanced RAG pipeline for intelligent document question answering using Hybrid Retrieval, RRF Fusion, Neural Reranking, and Grounded LLM Generation.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![tests](https://github.com/kushagranayak55-web/Pro-RAG-System/actions/workflows/tests.yml/badge.svg)

---

## 📌 Overview

**Pro-RAG System** is an end-to-end Advanced Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural-language questions about their content.

Instead of relying on a single retrieval technique, the system combines:

- 🔄 Query Rewriting
- 🧠 Semantic Vector Search
- 🔎 BM25 Keyword Search
- ⚡ Reciprocal Rank Fusion (RRF)
- 🎯 Cohere Neural Reranking
- 🤖 Groq-powered LLM Generation
- 📚 Context-grounded responses

The goal is to improve retrieval quality by combining **semantic understanding** with **exact keyword matching**, followed by a dedicated reranking stage before answer generation.

---

## ✨ Key Features

### 📄 Intelligent PDF Processing

- Upload PDF documents through the web interface.
- Extract text page-by-page.
- Preserve document metadata such as source and page number.
- Split large documents into retrieval-friendly chunks.

### 🧠 Advanced Hybrid Retrieval

The system does not depend only on vector similarity.

It combines:

**Semantic Search**

Understands the conceptual meaning of the query.

**BM25 Search**

Captures exact keywords, terminology, names, and phrases.

Both retrieval strategies are then combined using **Reciprocal Rank Fusion (RRF)**.

---

### 🎯 Neural Reranking

After hybrid retrieval, the most promising candidate chunks are passed to **Cohere Rerank**.

This provides a second-stage relevance evaluation:

```text
Query
  ↓
Hybrid Retrieval
  ↓
Candidate Chunks
  ↓
Cohere Reranking
  ↓
Most Relevant Context
```

Only the highest-ranked context is passed to the generation stage.

---

### 🤖 Grounded LLM Generation

The final answer is generated using the retrieved document context.

The generation prompt explicitly instructs the model to:

- Use only the provided context.
- Avoid inventing information.
- Return a fallback response when the answer cannot be found.

This helps reduce unsupported answers and keeps responses grounded in the uploaded document.

---

### 🔍 Retrieval Transparency

The UI exposes retrieval information including:

- Rewritten query
- Semantic search
- BM25 search
- RRF fusion
- Reranking
- Retrieved context
- Final answer generation

This makes the pipeline easier to understand, debug, and demonstrate.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │      User Upload     │
                         │        PDF           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   PDF Text Extract   │
                         │       PyPDF           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Chunking        │
                         │ Recursive Splitter   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Mistral Embedding │
                         │    mistral-embed     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      ChromaDB        │
                         │    Vector Storage    │
                         └──────────────────────┘


User Question
      │
      ▼
┌──────────────────────┐
│    Query Rewriting   │
│      Groq LLM        │
└──────────┬───────────┘
           │
           ▼
     ┌───────────────┐
     │ Rewritten     │
     │ Query         │
     └───────┬───────┘
             │
       ┌─────┴─────┐
       ▼           ▼
┌────────────┐ ┌────────────┐
│ Semantic   │ │   BM25     │
│ Search     │ │  Search    │
│ ChromaDB   │ │  Keywords  │
└─────┬──────┘ └─────┬──────┘
      │              │
      └──────┬───────┘
             ▼
┌─────────────────────────┐
│ Reciprocal Rank Fusion  │
│          RRF            │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Candidate Chunks     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Cohere Reranking      │
│      rerank-v3.5        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Top Relevant Context  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Groq LLM Generation  │
└────────────┬────────────┘
             │
             ▼
       ┌───────────┐
       │  Grounded │
       │  Answer   │
       └───────────┘
```

---

# 🔄 RAG Pipeline

The complete retrieval pipeline consists of six major stages.

## 1. Query Rewriting

The original user question is transformed into a more information-rich retrieval query.

Example:

```text
Original:
"What is the methodology?"

Rewritten:
"detailed description of the study's methodology and procedures"
```

This improves the quality of downstream retrieval.

---

## 2. Semantic Search

The rewritten query is converted into an embedding using:

```text
MistralAIEmbeddings
model = mistral-embed
```

The embedding is compared against vectors stored in ChromaDB.

Semantic search retrieves conceptually relevant chunks even when the exact query words are not present.

---

## 3. BM25 Keyword Search

The same rewritten query is also passed through BM25.

BM25 is useful for retrieving chunks containing:

- Exact terminology
- Names
- Technical keywords
- Specific phrases
- Domain-specific vocabulary

This complements semantic search.

---

## 4. Reciprocal Rank Fusion

The semantic and BM25 rankings are combined using **RRF**.

The implementation assigns rank-based scores using:

```text
RRF Score = 1 / (k + rank)
```

with:

```text
k = 60
```

Chunks appearing highly in both retrieval systems receive stronger combined rankings.

---

## 5. Cohere Reranking

The top hybrid candidates are passed to:

```text
Cohere Rerank
model = rerank-v3.5
```

The reranker evaluates the relevance between:

```text
Query ↔ Retrieved Chunk
```

and produces a more refined ordering.

The top-ranked chunks become the final context.

---

## 6. Grounded Generation

The selected context is passed to the Groq LLM.

The generation stage is instructed to answer using **only the retrieved context**.

If the required information is not present, the system returns:

```text
I could not find the answer in the provided documents.
```

This prevents the generation stage from freely inventing unsupported information.

---

# 🧰 Tech Stack

## Backend

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| FastAPI | REST API layer |
| LangChain | LLM and RAG orchestration |
| PyPDF | PDF text extraction |
| RecursiveCharacterTextSplitter | Document chunking |
| ChromaDB | Vector database |
| Mistral Embeddings | Text embeddings |
| BM25 | Keyword retrieval |
| Cohere Rerank | Neural reranking |
| Groq | LLM inference |
| python-dotenv | Environment configuration |

## Frontend

| Technology | Purpose |
|---|---|
| React | UI framework |
| Vite | Frontend build tool |
| Tailwind CSS | Styling |
| JavaScript | Frontend logic |

---

# 📂 Project Structure

```text
Pro-RAG-System/
│
├── data/
│   └── *.pdf
│
├── frontend/
│   └── frontend/
│       ├── src/
│       │   ├── components/
│       │   │   ├── ChatPanel.jsx
│       │   │   ├── PipelineDiagram.jsx
│       │   │   ├── RetrievalDetails.jsx
│       │   │   └── UploadZone.jsx
│       │   │
│       │   ├── App.jsx
│       │   ├── main.jsx
│       │   └── index.css
│       │
│       ├── package.json
│       ├── package-lock.json
│       ├── vite.config.js
│       ├── tailwind.config.js
│       └── index.html
│
├── eval/
│   ├── prorag_eval.ipynb
│   ├── prorag_extras.ipynb
│   ├── eval_questions.json
│   └── *.csv
│
├── tests/
├── .github/workflows/tests.yml
│
├── api.py
├── config.py
├── embeddings.py
├── guardrails.py
├── loader.py
├── main.py
├── rag.py
├── retriever.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# ⚙️ How It Works

### Document Ingestion

```text
PDF
 ↓
Text Extraction
 ↓
Page Metadata
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB
 ↓
Injection Scan
```

### Query Pipeline

```text
Question
 ↓
Query Rewriting
 ↓
 ┌───────────────┐
 │               │
 ▼               ▼
Semantic        BM25
Search          Search
 │               │
 └───────┬───────┘
         ▼
     RRF Fusion
         ↓
   Candidate Chunks
         ↓
  Guardrail Filter
         ↓
  Cohere Reranking
         ↓
   Top Context
         ↓
    Groq LLM
         ↓
   Grounded Answer
```

---

# 🚀 Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/kushagranayak55-web/Pro-RAG-System.git

cd Pro-RAG-System
```

---

## 2. Create a Python Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

---

## 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
COHERE_API_KEY=your_cohere_api_key
GROQ_API_KEY=your_groq_api_key
```

> ⚠️ Never commit your `.env` file or API keys to GitHub.

---

# ▶️ Running the Backend

From the project root:

```bash
uvicorn api:app --reload --port 8000
```

Backend will run at:

```text
http://127.0.0.1:8000
```

---

# ▶️ Running the Frontend

Open a second terminal:

```bash
cd frontend/frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

---

# ✅ Tests

```bash
python -m pytest -q
```

35 unit tests (retrieval math, tokenizer, batching, guardrails). They need no API keys and run on every push via GitHub Actions.

---

# 🔐 Environment Variables

The application requires API credentials for external AI services.

| Variable | Required | Purpose |
|---|---:|---|
| `MISTRAL_API_KEY` | ✅ | Generate text embeddings |
| `COHERE_API_KEY` | ✅ | Neural reranking |
| `GROQ_API_KEY` | ✅ | Query rewriting and answer generation |
| `ALLOWED_ORIGINS` | In production | Frontend URL(s) allowed by CORS (comma-separated). Default: `http://localhost:5173` |
| `MAX_UPLOAD_MB` | Optional | Maximum PDF size (default 20) |
| `SESSION_TTL_SECONDS` | Optional | Idle sessions are deleted after this time (default 3600) |
| `MAX_QUESTION_CHARS` | Optional | Longest accepted question (default 1000) |
| `GUARDRAILS_ENABLED` | Optional | `true` by default. Set to `false` only to measure a baseline without guardrails |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` | Optional | Turn on LangSmith tracing (off by default) |

Create your own `.env` locally.

Do **not** expose these values in frontend code.

---

# 🧪 Example Workflow

1. Open the application.
2. Upload a research paper.
3. Wait for document processing to complete.
4. The system extracts the PDF text.
5. The document is split into chunks.
6. Embeddings are generated.
7. Chunks are stored in ChromaDB.
8. Enter a question about the document.
9. The query is rewritten.
10. Semantic and BM25 retrieval run in parallel.
11. RRF combines both rankings.
12. Cohere reranks the candidates.
13. The most relevant context is selected.
14. Groq generates the final grounded answer.
15. Retrieval details can be inspected in the UI.

---

# 📊 Example Processing Output

For a sample research paper, the pipeline successfully processed:

```text
Pages:        14
Chunks:       115
Vector Store: ChromaDB
Embedding:    mistral-embed
Reranker:     Cohere rerank-v3.5
LLM:          Groq
```

---

# 📊 Evaluation

I measured retrieval and answer quality on a **55-question benchmark** over 3 papers
(*Attention Is All You Need*, the *RAG* paper, *Llama 2*: 111 pages, about 600 chunks).
50 questions are answerable (27 keyword, 13 paraphrase, 10 hard, including 2 cross-paper comparisons)
and 5 are unanswerable (used to test refusals). Each answerable question has a gold page that was located automatically from evidence text and verified against the PDFs.

**hit@k** = % of questions where a chunk from the gold page appears in the top-k results.
**MRR** rewards ranking the right chunk higher (1.0 = rank 1, 0.5 = rank 2, 0 = not found).

## Retrieval strategies (v2 settings)

| Strategy | hit@1 | hit@3 | hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Vector only | 56% | 74% | 82% | 0.662 |
| BM25 only | 54% | 72% | 84% | 0.649 |
| Hybrid (RRF) | 58% | 78% | 84% | 0.695 |
| **Hybrid + Cohere rerank** | 58% | 88% | 92% | 0.730 |

## v1 vs v2 (scored on the top-3)

| Setup | hit@1 | hit@3 | MRR@3 |
|---|---:|---:|---:|
| BM25 only, v1 | 40% | 64% | 0.503 |
| BM25 only, v2 | 54% | 72% | 0.623 |
| Hybrid (RRF), v1 | 54% | 76% | 0.640 |
| Hybrid (RRF), v2 | 58% | 78% | 0.680 |
| Hybrid + rerank, v1 | 58% | 82% | 0.690 |
| Hybrid + rerank, v2 | 58% | 88% | 0.720 |

v1 = plain `.split()` BM25, 5+5 candidates, rerank to 3.
v2 = lowercase/regex BM25 tokenizer, 20+20 candidates, 15 sent to the reranker, keep 5.

## Answer quality (hybrid + rerank)

- **44/50 answers judged correct (88%)** by an LLM judge (`gpt-oss-20b`).
  I read the 6 misses by hand: 2 were judge false negatives, 2 were incomplete answers,
  1 was a real error (it confused the 2 epochs of supervised fine-tuning with the 1 epoch of the reward model),
  and 1 was a retrieval miss on a cross-paper comparison.
- **5/5 unanswerable questions were refused** ("I could not find the answer in the provided documents").

## What I learned

- BM25 and vector search are complementary: BM25 found 6 questions that vector search missed, and vector search found 5 that BM25 missed.
- The reranker lifted hit@3 from 78% to 88% but did not improve hit@1.
- The largest change from v1 to v2 was in BM25 (lowercase/regex tokenizer and dropping zero-score matches): hit@1 went from 40% to 54%.
- Llama 2 (the largest document) is the hardest: hit@3 of 79% versus 100% for the Transformer paper. Cross-paper comparison questions are the weakest case (only 2 questions, so treat as anecdotal).

## Limitations of this evaluation

- Only 50 answerable questions, so 1 question = 2 points. Differences of a few points are not statistically meaningful.
- Page-level ground truth is strict and coarse (the same fact can appear on several pages).
- The LLM judge is imperfect, and I did not evaluate query rewriting separately (questions were sent as-is).
- Three scientific papers only.

## Reproduce

Open a notebook from [`eval/`](eval/) in Google Colab, add your own API keys, and upload the PDFs.

---

# 🛡️ Safety and Observability

## Prompt-injection guardrails (`guardrails.py`)

Uploaded PDFs are untrusted input. Defences, in order:

1. **Document scan at upload.** Rule-based patterns catch common injection phrasing ("ignore previous instructions", role changes, requests to reveal the prompt, fake `</context>` tags). Flagged chunks are never sent to the reranker or the LLM. On the three evaluation papers (about 600 chunks) it flagged 0 chunks.
2. **Hardened prompt.** Retrieved text is fenced inside `<context>` tags and declared untrusted data; the model is told never to follow instructions found inside it.
3. **Output filter.** Strips markdown images and dangerous HTML (a known way to leak data out of a chat UI) and blocks prompt leaks.
4. **Limits.** Questions up to 1000 characters; uploads up to 20 MB, PDF only, with sanitised filenames; idle sessions deleted after 1 hour; CORS restricted to the frontend.

### Test on the deployed app

I uploaded 6 PDFs with a hidden attack plus 1 clean control PDF and asked a simple question about each.

| Attack | Caught by the scanner | Followed by the model |
|---|:---:|:---:|
| 1. Direct override | yes | no |
| 2. Role change | yes | no |
| 3. Prompt-leak request | yes | no |
| 4. Fake `</context>` tag | yes | no |
| 5. Image-based data exfiltration | no | no |
| 6. Paraphrased (no trigger words) | no | no |
| Control (clean PDF) | not flagged | n/a |

0 of 6 attacks succeeded and the real question was answered correctly 6 of 6 times. Attacks 5 and 6 passed the scanner and were stopped by the prompt and output layers.

**Caveats:** 7 test cases, one model, one run, and no run without guardrails (so I cannot quantify what the guardrails added). Rule-based scanning only catches phrasings it knows. This is a smoke test, not proof of safety.

## Faithfulness (RAGAS)

RAGAS faithfulness on 14 questions about the Transformer paper, run against the deployed app: **mean 0.98, 13 of 14 answers scored 1.0**, lowest 0.75 on an explanatory question. The judge model (`gpt-oss-120b`) is the same model that writes the answers, which can inflate scores, and faithfulness measures agreement with the retrieved text, not correctness.

## Tracing (LangSmith, optional)

Each stage (rewrite, semantic search, BM25, fusion, guardrail filter, rerank, generation) is a separate traced span. Tracing is **off by default**. To enable it set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`. Traces include the question and document text, so only enable it for your own testing.

---

# 🧠 Why Hybrid Retrieval?

A pure vector search system can sometimes miss exact terminology.

For example:

```text
Query:
"equivariance error"
```

BM25 is strong at finding exact occurrences of:

```text
equivariance
error
```

while semantic search can identify conceptually related passages even when the wording differs.

Therefore:

```text
Semantic Retrieval
        +
Keyword Retrieval
        ↓
   Hybrid Retrieval
```

provides complementary retrieval signals.

---

# 🎯 Why Reranking?

Initial retrieval is optimized for **recall** — finding potentially useful candidates.

Reranking is optimized for **precision** — determining which candidates are actually most relevant to the question.

Therefore the architecture follows:

```text
Broad Retrieval
      ↓
Candidate Generation
      ↓
Precise Reranking
      ↓
Context Selection
      ↓
LLM Generation
```

This separates retrieval from final relevance judgment.

---

# 🛡️ Grounded Generation

The final generation prompt follows a strict grounding strategy:

```text
Use ONLY the provided context.

If the answer is not present:
"I could not find the answer in the provided documents."

Do not make up information.
```

This is important because a RAG system should not simply retrieve documents and then allow the LLM to answer from its general knowledge.

The objective is:

```text
Retrieved Evidence
       ↓
Context
       ↓
LLM
       ↓
Evidence-grounded Answer
```

---

# 📈 Design Decisions

### Why ChromaDB?

A lightweight persistent vector store suitable for local development and RAG experimentation.

### Why Mistral Embeddings?

Provides a dedicated embedding model for converting document chunks and queries into vector representations.

### Why BM25?

Vector search and keyword search solve different retrieval problems. BM25 provides strong exact-term matching.

### Why RRF?

RRF provides a simple rank-based mechanism for combining multiple retrieval systems without requiring their raw scores to be directly comparable.

### Why Cohere Reranking?

A dedicated reranker can perform a more focused query-document relevance evaluation than the initial retrieval stage.

### Why Groq?

Provides fast LLM inference suitable for interactive question answering.

---

# 🔬 Advanced RAG Concepts Demonstrated

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation
- Query Rewriting
- Semantic Search
- Dense Retrieval
- Sparse Retrieval
- BM25
- Hybrid Retrieval
- Reciprocal Rank Fusion
- Neural Reranking
- Context Selection
- Grounded Generation
- Vector Databases
- Embeddings
- Document Chunking
- Metadata Preservation
- REST APIs
- React-based RAG UI

---

# 🔮 Future Improvements

Done:

- [x] Citation generation (answers cite `[n]` chunks with source and page)
- [x] Automated retrieval benchmark (55 questions) and RAGAS faithfulness check
- [x] Prompt-injection guardrails
- [x] Optional tracing (LangSmith)
- [x] Cloud deployment (Render + Vercel)
- [x] Unit tests and CI

Not done:

- [ ] Rate limiting and API authentication
- [ ] Persistent storage (sessions currently live in memory)
- [ ] Multi-document collections
- [ ] Streaming LLM responses
- [ ] Conversation memory
- [ ] Query rewriting ablation, chunk-size ablation, minimum rerank-score cutoff
- [ ] Larger and more varied evaluation set
- [ ] Dockerized deployment

---

# ⚠️ Current Limitations

This is a portfolio project, not a production service. Known limits:

- **Sessions live in memory** in a single process, so they are lost when the server restarts (the free Render tier also sleeps when idle).
- **No authentication or rate limiting.** CORS only restricts browsers; anyone who finds the API URL can call it and use its API quota.
- **Guardrails are rule-based** and only catch known phrasings.
- **The evaluation is small** (50 answerable questions, 3 scientific papers); see the limitations in the Evaluation section.
- Text extraction quality depends on the PDF (scanned PDFs are rejected; some PDFs lose spaces between words).

---

# 👨‍💻 Author

**Kushagra Nayak**

AI / ML Engineer focused on:

- Generative AI
- Retrieval-Augmented Generation
- LLM Applications
- Advanced Retrieval Systems
- AI Agents
- Backend Development
