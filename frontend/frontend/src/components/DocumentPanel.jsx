import PipelineDiagram from "./PipelineDiagram.jsx";

function Stat({ label, value }) {
  return (
    <div className="rounded-lg border border-border bg-elevated px-3 py-2.5">
      <div className="font-mono text-lg font-medium text-ink">{value}</div>
      <div className="text-[11px] text-ink-muted mt-0.5">{label}</div>
    </div>
  );
}

export default function DocumentPanel({ doc, onUploadNewClick, onClearConversation, hasMessages }) {
  return (
    <div className="space-y-5">
      {/* Document card */}
      <div className="rounded-2xl border border-border bg-surface p-5 shadow-card">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent-blue/10 border border-accent-blue/20">
            <svg viewBox="0 0 24 24" className="h-5 w-5 text-accent-blue">
              <path fill="currentColor" d="M6 2h9l5 5v15a1 1 0 01-1 1H6a1 1 0 01-1-1V3a1 1 0 011-1zm8 1.5V8h4.5L14 3.5z" />
            </svg>
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-ink" title={doc.filename}>
              {doc.filename}
            </p>
            <div className="mt-1 flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-accent-green" />
              <span className="text-xs text-accent-green">Ready</span>
            </div>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-3 gap-2">
          <Stat label="Pages" value={doc.num_pages} />
          <Stat label="Chunks" value={doc.num_chunks} />
          <Stat label="Size" value={`${doc.file_size_mb} MB`} />
        </div>

        <div className="mt-4 flex gap-2">
          <button
            onClick={onUploadNewClick}
            className="flex-1 rounded-lg border border-border bg-elevated px-3 py-2 text-xs font-medium text-ink transition-colors hover:border-accent-blue/40 hover:text-accent-blue"
          >
            + Upload New PDF
          </button>
          {hasMessages ? (
            <button
              onClick={onClearConversation}
              className="rounded-lg border border-border bg-elevated px-3 py-2 text-xs font-medium text-ink-muted transition-colors hover:border-accent-red/40 hover:text-accent-red"
              title="Clear conversation"
            >
              Clear
            </button>
          ) : null}
        </div>
      </div>

      {/* Static pipeline architecture */}
      <div className="rounded-2xl border border-border bg-surface p-5 shadow-card">
        <p className="mb-4 text-xs font-medium uppercase tracking-wider text-ink-muted">RAG Pipeline</p>
        <PipelineDiagram mode="static" />
      </div>
    </div>
  );
}
