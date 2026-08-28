import { useEffect, useRef, useState } from "react";
import UploadZone from "./components/UploadZone.jsx";
import DocumentPanel from "./components/DocumentPanel.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import PipelineDiagram from "./components/PipelineDiagram.jsx";
import { uploadDocument, askQuestion, deleteSession } from "./api.js";

const HOW_IT_WORKS = [
  { num: "01", title: "Query Rewriting", text: "Turns your question into a retrieval-friendly query." },
  { num: "02", title: "Semantic Search", text: "Finds conceptually relevant chunks in the vector store." },
  { num: "03", title: "BM25 Search", text: "Finds exact keyword-based matches in the document." },
  { num: "04", title: "RRF Fusion", text: "Combines semantic and keyword rankings into one list." },
  { num: "05", title: "Cohere Reranking", text: "Reorders candidates by true relevance to your question." },
  { num: "06", title: "LLM Generation", text: "Generates a grounded answer from the selected context." },
];

const INGEST_STEPS = [
  { key: "loaded", label: "PDF Loaded" },
  { key: "extracted", label: "Text Extraction" },
  { key: "chunked", label: "Chunking" },
  { key: "embedded", label: "Generating Embeddings" },
  { key: "indexed", label: "Building Vector Index" },
  { key: "ready", label: "Ready" },
];

function Nav() {
  return (
    <header className="sticky top-0 z-20 border-b border-border-soft bg-base/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <div className="flex items-center gap-2">
          <span className="text-lg">🧠</span>
          <span className="font-display text-[15px] font-semibold text-ink">ProRAG System</span>
        </div>
        <nav className="hidden gap-7 text-sm text-ink-muted sm:flex">
          <a href="#how-it-works" className="transition-colors hover:text-ink">How it Works</a>
          <a href="#rag-pipeline" className="transition-colors hover:text-ink">RAG Pipeline</a>
          <span className="cursor-default text-ink-faint">GitHub</span>
        </nav>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="border-t border-border-soft px-6 py-10 text-center">
      <p className="font-display text-sm font-medium text-ink">ProRAG System</p>
      <p className="mt-1.5 text-xs text-ink-muted">
        Advanced Retrieval • Hybrid Search • Reranking • Grounded Generation
      </p>
      <p className="mt-3 text-[11px] text-ink-faint">Engineered by Kushagra Nayak</p>
    </footer>
  );
}

function ProcessingScreen({ filename, revealedCount, result, error, onRetry }) {
  return (
    <div className="mx-auto max-w-md px-6 py-24">
      <div className="rounded-2xl border border-border bg-surface p-6 shadow-card">
        {error ? (
          <>
            <p className="mb-1 font-display text-base font-medium text-accent-red">Unable to process this document</p>
            <p className="mb-5 text-sm text-ink-muted">{error}</p>
            <button
              onClick={onRetry}
              className="w-full rounded-lg border border-accent-blue/30 bg-accent-blue/10 px-4 py-2 text-sm font-medium text-accent-blue hover:bg-accent-blue/20"
            >
              Try Again
            </button>
          </>
        ) : (
          <>
            <p className="mb-1 font-display text-base font-medium text-ink">Processing your document</p>
            <p className="mb-5 truncate text-sm text-ink-muted">{filename}</p>

            <div className="space-y-2.5">
              {INGEST_STEPS.map((step, i) => {
                const state = i < revealedCount ? "complete" : i === revealedCount ? "running" : "waiting";
                return (
                  <div key={step.key} className="flex items-center gap-3">
                    <span
                      className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px]
                      ${state === "complete" ? "bg-accent-green/15 text-accent-green" : ""}
                      ${state === "running" ? "bg-accent-blue/15 text-accent-blue animate-pulseDot" : ""}
                      ${state === "waiting" ? "bg-elevated text-ink-faint border border-border" : ""}`}
                    >
                      {state === "complete" ? "✓" : state === "running" ? "●" : "○"}
                    </span>
                    <span className={`text-sm ${state === "waiting" ? "text-ink-faint" : "text-ink"}`}>
                      {step.label}
                    </span>
                  </div>
                );
              })}
            </div>

            {result && revealedCount >= INGEST_STEPS.length ? (
              <div className="mt-5 grid grid-cols-2 gap-2 border-t border-border pt-4 font-mono text-xs text-ink-muted">
                <span>{result.num_pages} pages</span>
                <span>{result.num_chunks} chunks</span>
              </div>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [doc, setDoc] = useState(null); // { session_id, filename, num_pages, num_chunks, file_size_mb }
  const [uploading, setUploading] = useState(false);
  const [uploadFilename, setUploadFilename] = useState("");
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadError, setUploadError] = useState("");
  const [revealedCount, setRevealedCount] = useState(0);

  const [messages, setMessages] = useState([]);
  const [isAsking, setIsAsking] = useState(false);
  const [liveStatuses, setLiveStatuses] = useState({});
  const [liveStats, setLiveStats] = useState({});
  const [showUploadModal, setShowUploadModal] = useState(false);
  const pendingFileRef = useRef(null);

  // Reveal ingestion steps one by one once the real result is known.
  useEffect(() => {
    if (!uploadResult) return;
    if (revealedCount >= INGEST_STEPS.length) {
      const t = setTimeout(() => {
        setDoc(uploadResult);
        setMessages([]);
        setUploading(false);
        setUploadResult(null);
        setRevealedCount(0);
        setShowUploadModal(false);
      }, 400);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => setRevealedCount((c) => c + 1), 320);
    return () => clearTimeout(t);
  }, [uploadResult, revealedCount]);

  async function handleFile(file) {
    pendingFileRef.current = file;
    setUploadFilename(file.name);
    setUploadError("");
    setUploadResult(null);
    setRevealedCount(0);
    setUploading(true);

    // Clean up the previous session so document contexts never mix.
    if (doc?.session_id) {
      deleteSession(doc.session_id);
    }

    try {
      const result = await uploadDocument(file);
      setUploadResult(result);
    } catch (e) {
      setUploadError(e.message);
    }
  }

  function retryUpload() {
    if (pendingFileRef.current) handleFile(pendingFileRef.current);
    else setUploading(false);
  }

  function handleClearConversation() {
    setMessages([]);
  }

  async function handleSend(question) {
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setIsAsking(true);
    setLiveStatuses({});
    setLiveStats({});

    try {
      const res = await askQuestion(doc.session_id, question);

      // Reveal the (already real) pipeline results in sequence for readability.
      const order = [
        ["query_rewriting", null],
        ["semantic_search", { semantic: res.pipeline.semantic_search.matches }],
        ["bm25_search", { bm25: res.pipeline.bm25_search.matches }],
        ["rrf_fusion", { candidates: res.pipeline.rrf_fusion.candidates }],
        ["reranking", { selected: res.pipeline.reranking.selected }],
        ["generation", null],
      ];

      for (const [key, statBump] of order) {
        await new Promise((r) => setTimeout(r, 260));
        setLiveStatuses((prev) => ({ ...prev, [key]: "complete" }));
        if (statBump) setLiveStats((prev) => ({ ...prev, ...statBump }));
      }

      await new Promise((r) => setTimeout(r, 200));

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          pipeline: res.pipeline,
          rewrittenQuery: res.rewritten_query,
          chunks: res.reranked_chunks,
        },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: e.message || "Something went wrong while generating your answer.", error: true },
      ]);
    } finally {
      setIsAsking(false);
      setLiveStatuses({});
      setLiveStats({});
    }
  }

  const isProcessingUpload = uploading || uploadError;

  return (
    <div className="flex min-h-screen flex-col">
      <Nav />

      <main className="flex-1">
        {isProcessingUpload ? (
          <ProcessingScreen
            filename={uploadFilename}
            revealedCount={revealedCount}
            result={uploadResult}
            error={uploadError}
            onRetry={retryUpload}
          />
        ) : !doc ? (
          <div className="mx-auto max-w-4xl px-6 py-20">
            {/* Hero */}
            <div className="mx-auto max-w-2xl text-center">
              <h1 className="font-display text-4xl font-semibold leading-tight text-ink sm:text-5xl">
                Chat with your documents.
                <br />
                <span className="bg-gradient-to-r from-accent-blue to-accent-purple bg-clip-text text-transparent">
                  Powered by Advanced RAG.
                </span>
              </h1>
              <p className="mt-5 text-[15px] leading-7 text-ink-muted">
                Upload a research paper and get grounded answers using hybrid retrieval,
                intelligent reranking, and LLM-powered generation.
              </p>
            </div>

            <div className="mx-auto mt-10 max-w-xl">
              <UploadZone onFile={handleFile} disabled={uploading} />
            </div>

            {/* Feature cards */}
            <div className="mt-16 grid gap-4 sm:grid-cols-3">
              {[
                { icon: "🔎", title: "Hybrid Retrieval", text: "Combines semantic vector search with BM25 keyword retrieval." },
                { icon: "🎯", title: "Smart Reranking", text: "Cohere reranks retrieved chunks to surface what actually matters." },
                { icon: "⚡", title: "Fast, Grounded Answers", text: "Groq generates concise answers grounded in your document." },
              ].map((f) => (
                <div key={f.title} className="rounded-2xl border border-border bg-surface p-5 shadow-card">
                  <div className="mb-3 text-xl">{f.icon}</div>
                  <p className="font-display text-sm font-medium text-ink">{f.title}</p>
                  <p className="mt-1.5 text-[13px] leading-5 text-ink-muted">{f.text}</p>
                </div>
              ))}
            </div>

            {/* How it works */}
            <div id="how-it-works" className="mt-24 scroll-mt-24">
              <p className="text-center text-xs font-medium uppercase tracking-wider text-ink-muted">
                How ProRAG Works
              </p>
              <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {HOW_IT_WORKS.map((s) => (
                  <div key={s.num} className="rounded-xl border border-border bg-surface p-4">
                    <span className="font-mono text-xs text-accent-blue">{s.num}</span>
                    <p className="mt-1.5 font-display text-sm font-medium text-ink">{s.title}</p>
                    <p className="mt-1 text-[13px] leading-5 text-ink-muted">{s.text}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Architecture diagram */}
            <div id="rag-pipeline" className="mx-auto mt-24 max-w-md scroll-mt-24">
              <p className="text-center text-xs font-medium uppercase tracking-wider text-ink-muted mb-6">
                Pipeline Architecture
              </p>
              <div className="rounded-2xl border border-border bg-surface p-5 shadow-card">
                <PipelineDiagram mode="static" />
              </div>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-6xl px-6 py-8">
            <div className="grid gap-5 lg:grid-cols-[300px_1fr]" style={{ height: "calc(100vh - 140px)" }}>
              <div className="overflow-y-auto pr-1">
                <DocumentPanel
                  doc={doc}
                  onUploadNewClick={() => setShowUploadModal(true)}
                  onClearConversation={handleClearConversation}
                  hasMessages={messages.length > 0}
                />
              </div>
              <ChatPanel
                messages={messages}
                onSend={handleSend}
                isProcessing={isAsking}
                liveStatuses={liveStatuses}
                liveStats={liveStats}
              />
            </div>
          </div>
        )}
      </main>

      <Footer />

      {showUploadModal ? (
        <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/60 px-6 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-2xl border border-border bg-surface p-6 shadow-card">
            <div className="mb-4 flex items-center justify-between">
              <p className="font-display text-sm font-medium text-ink">Upload a new PDF</p>
              <button
                onClick={() => setShowUploadModal(false)}
                className="text-ink-faint hover:text-ink"
              >
                ✕
              </button>
            </div>
            <p className="mb-4 text-xs text-ink-muted">
              This replaces the current document and starts a new conversation.
            </p>
            <UploadZone onFile={handleFile} disabled={uploading} />
          </div>
        </div>
      ) : null}
    </div>
  );
}
