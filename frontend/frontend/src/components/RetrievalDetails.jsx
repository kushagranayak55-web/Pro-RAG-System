import { useState } from "react";

export default function RetrievalDetails({ pipeline, rewrittenQuery, chunks }) {
  const [open, setOpen] = useState(false);
  const [expandedChunk, setExpandedChunk] = useState(null);

  const badges = [
    { label: "Semantic Search", ok: pipeline.semantic_search?.status === "complete" },
    { label: "BM25 Search", ok: pipeline.bm25_search?.status === "complete" },
    { label: "RRF Fusion", ok: pipeline.rrf_fusion?.status === "complete" },
    { label: "Cohere Reranking", ok: pipeline.reranking?.status === "complete" },
  ];

  return (
    <div className="mt-3 rounded-xl border border-border bg-surface/60">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-left"
      >
        <span className="text-xs font-medium text-ink-muted">Retrieval Details</span>
        <svg
          viewBox="0 0 20 20"
          className={`h-4 w-4 text-ink-faint transition-transform ${open ? "rotate-180" : ""}`}
        >
          <path fill="currentColor" d="M5.8 7.5h8.4L10 12.5z" />
        </svg>
      </button>

      {open ? (
        <div className="border-t border-border px-4 py-4 animate-riseIn">
          {rewrittenQuery ? (
            <p className="mb-3 text-xs text-ink-faint">
              <span className="text-ink-muted">Rewritten query: </span>
              <span className="font-mono">{rewrittenQuery}</span>
            </p>
          ) : null}

          <div className="mb-4 flex flex-wrap gap-2">
            {badges.map((b) => (
              <span
                key={b.label}
                className="inline-flex items-center gap-1.5 rounded-full border border-accent-green/30 bg-accent-green/10 px-2.5 py-1 text-[11px] text-accent-green"
              >
                <span className="h-1 w-1 rounded-full bg-accent-green" />
                {b.label}
              </span>
            ))}
          </div>

          <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-ink-muted">
            Retrieved Context
          </p>

          <div className="space-y-2">
            {chunks.map((chunk, i) => (
              <div key={i} className="rounded-lg border border-border bg-elevated p-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-xs font-medium text-ink-muted">Context #{i + 1}</span>
                  <div className="flex items-center gap-2">
                    {chunk.score != null ? (
                      <div className="flex items-center gap-1.5">
                        <div className="h-1 w-16 overflow-hidden rounded-full bg-raised">
                          <div
                            className="h-full rounded-full bg-accent-blue"
                            style={{ width: `${Math.min(chunk.score * 100, 100)}%` }}
                          />
                        </div>
                        <span className="font-mono text-[11px] text-ink-muted">{chunk.score.toFixed(3)}</span>
                      </div>
                    ) : null}
                  </div>
                </div>

                <p
                  className={`mt-2 text-xs leading-5 text-ink-muted ${
                    expandedChunk === i ? "" : "line-clamp-2"
                  }`}
                >
                  {chunk.text}
                </p>

                <button
                  onClick={() => setExpandedChunk(expandedChunk === i ? null : i)}
                  className="mt-1.5 text-[11px] text-accent-blue hover:underline"
                >
                  {expandedChunk === i ? "Show less" : "Show more"}
                </button>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
