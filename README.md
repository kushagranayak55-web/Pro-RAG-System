# 🚀 Pro-RAG System

> **Production-oriented Advanced RAG pipeline for intelligent document question answering using Hybrid Retrieval, RRF Fusion, Neural Reranking, and Grounded LLM Generation.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

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
├── api.py
├── config.py
├── embeddings.py
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

# 🔐 Environment Variables

The application requires API credentials for external AI services.

| Variable | Required | Purpose |
|---|---:|---|
| `MISTRAL_API_KEY` | ✅ | Generate text embeddings |
| `COHERE_API_KEY` | ✅ | Neural reranking |
| `GROQ_API_KEY` | ✅ | Query rewriting and answer generation |

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

Potential improvements include:

- [ ] Multi-document collections
- [ ] Conversation memory
- [ ] Streaming LLM responses
- [ ] Metadata-aware filtering
- [ ] Parent-child retrieval
- [ ] Multi-query retrieval
- [ ] HyDE retrieval
- [ ] Query decomposition
- [ ] Citation generation
- [ ] Retrieval evaluation using RAGAS
- [ ] Automated retrieval benchmarks
- [ ] Authentication and user accounts
- [ ] Production vector database
- [ ] Dockerized deployment
- [ ] Cloud deployment
- [ ] Observability and tracing
- [ ] Rate limiting and API security

---

# ⚠️ Current Limitations

This project is primarily designed as an advanced RAG demonstration and portfolio project.

For a large-scale production environment, additional considerations would include:

- Distributed vector storage
- Persistent document management
- Authentication
- API rate limiting
- Background document processing
- Caching
- Monitoring
- Evaluation pipelines
- Horizontal scaling
- Secure file handling

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
