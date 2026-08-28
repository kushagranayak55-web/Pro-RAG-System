const STAGES = [
  { key: "query_rewriting", num: "01", label: "Query Rewriting", branch: false },
  { key: "retrieval", num: "02", label: "Semantic Search  +  BM25 Search", branch: true },
  { key: "rrf_fusion", num: "03", label: "RRF Hybrid Fusion", branch: false },
  { key: "reranking", num: "04", label: "Cohere Reranking", branch: false },
  { key: "generation", num: "05", label: "Groq LLM Generation", branch: false },
];

// status: "waiting" | "running" | "complete"
function Node({ num, label, status, detail }) {
  const ring =
    status === "complete"
      ? "border-accent-green/50 bg-accent-green/10"
      : status === "running"
      ? "border-accent-blue/60 bg-accent-blue/10"
      : "border-border bg-elevated";

  const dot =
    status === "complete"
      ? "bg-accent-green"
      : status === "running"
      ? "bg-accent-blue animate-pulseDot"
      : "bg-ink-faint";

  const numColor =
    status === "complete"
      ? "text-accent-green"
      : status === "running"
      ? "text-accent-blue"
      : "text-ink-faint";

  return (
    <div className={`flex items-center gap-3 rounded-xl border px-4 py-3 transition-colors duration-300 ${ring}`}>
      <span className={`font-mono text-[11px] tracking-wider ${numColor}`}>{num}</span>
      <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${dot}`} />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-ink truncate">{label}</div>
        {detail ? <div className="text-[11px] font-mono text-ink-muted mt-0.5">{detail}</div> : null}
      </div>
      {status === "complete" ? (
        <svg viewBox="0 0 20 20" className="h-4 w-4 text-accent-green shrink-0">
          <path
            fill="currentColor"
            d="M8.2 13.6 4.9 10.3l1.4-1.4 1.9 1.9 5.5-5.5 1.4 1.4z"
          />
        </svg>
      ) : null}
    </div>
  );
}

function Connector({ active }) {
  return (
    <div className="flex justify-center py-1">
      <div className={`w-px h-4 ${active ? "bg-accent-blue/60" : "bg-border"}`} />
    </div>
  );
}

/**
 * PipelineDiagram
 *
 * mode="static"  -> always shows the architecture, all nodes neutral (landing page explainer)
 * mode="live"    -> `statuses` drives real waiting/running/complete states during a live question
 * statuses: { query_rewriting, semantic_search, bm25_search, rrf_fusion, reranking, generation }
 * stats: optional real counts, e.g. { semantic: 5, bm25: 5, candidates: 8, selected: 3 }
 */
export default function PipelineDiagram({ mode = "static", statuses = {}, stats = {} }) {
  const get = (key) => (mode === "static" ? "waiting" : statuses[key] || "waiting");

  return (
    <div className="space-y-0">
      <Node num="•" label="Your Question" status={mode === "live" ? "complete" : "waiting"} />
      <Connector active={get("query_rewriting") !== "waiting"} />
      <Node num="01" label="Query Rewriting" status={get("query_rewriting")} />
      <Connector active={get("semantic_search") !== "waiting" || get("bm25_search") !== "waiting"} />

      <div className="grid grid-cols-2 gap-2">
        <Node
          num="02"
          label="Semantic Search"
          status={get("semantic_search")}
          detail={stats.semantic != null ? `${stats.semantic} matches` : undefined}
        />
        <Node
          num="03"
          label="BM25 Search"
          status={get("bm25_search")}
          detail={stats.bm25 != null ? `${stats.bm25} matches` : undefined}
        />
      </div>

      <Connector active={get("rrf_fusion") !== "waiting"} />
      <Node
        num="04"
        label="RRF Hybrid Fusion"
        status={get("rrf_fusion")}
        detail={stats.candidates != null ? `${stats.candidates} candidates` : undefined}
      />
      <Connector active={get("reranking") !== "waiting"} />
      <Node
        num="05"
        label="Cohere Reranking"
        status={get("reranking")}
        detail={stats.selected != null ? `top ${stats.selected} selected` : undefined}
      />
      <Connector active={get("generation") !== "waiting"} />
      <Node num="06" label="Groq LLM Generation" status={get("generation")} />
      <Connector active={get("generation") === "complete"} />
      <Node num="•" label="Final Answer" status={get("generation") === "complete" ? "complete" : "waiting"} />
    </div>
  );
}

export { STAGES };
