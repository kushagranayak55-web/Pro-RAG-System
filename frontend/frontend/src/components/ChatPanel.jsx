import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import PipelineDiagram from "./PipelineDiagram.jsx";
import RetrievalDetails from "./RetrievalDetails.jsx";

const SUGGESTIONS = [
  "What is the main contribution of this research?",
  "Explain the methodology.",
  "What are the key findings?",
  "Summarize this research.",
  "What problem does this paper address?",
];

function UserBubble({ content }) {
  return (
    <div className="flex justify-end animate-riseIn">
      <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-accent-blue/15 border border-accent-blue/20 px-4 py-2.5 text-sm text-ink">
        {content}
      </div>
    </div>
  );
}

function AssistantBubble({ message }) {
  return (
    <div className="flex justify-start animate-riseIn">
      <div className="max-w-[85%] w-full">
        <div className="mb-1.5 flex items-center gap-1.5">
          <span className="text-sm">🧠</span>
          <span className="text-xs font-medium text-ink-muted">ProRAG System</span>
        </div>

        {message.error ? (
          <div className="rounded-xl border border-accent-red/30 bg-accent-red/10 px-4 py-3 text-sm text-accent-red">
            {message.content}
          </div>
        ) : (
          <div className="rounded-xl rounded-tl-sm border border-border bg-surface px-4 py-3">
            <div className="answer-prose">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          </div>
        )}

        {message.pipeline && !message.error ? (
          <RetrievalDetails
            pipeline={message.pipeline}
            rewrittenQuery={message.rewrittenQuery}
            chunks={message.chunks || []}
          />
        ) : null}
      </div>
    </div>
  );
}

function LiveProcessing({ statuses, stats }) {
  return (
    <div className="flex justify-start animate-riseIn">
      <div className="w-full max-w-[85%] rounded-xl border border-border bg-surface px-4 py-4">
        <p className="mb-3 text-xs font-medium text-ink-muted">Analyzing your question…</p>
        <PipelineDiagram mode="live" statuses={statuses} stats={stats} />
      </div>
    </div>
  );
}

export default function ChatPanel({ messages, onSend, isProcessing, liveStatuses, liveStats }) {
  const [input, setInput] = useState("");
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages.length, isProcessing, liveStatuses]);

  function submit(text) {
    const q = (text ?? input).trim();
    if (!q || isProcessing) return;
    onSend(q);
    setInput("");
  }

  return (
    <div className="flex h-full flex-col rounded-2xl border border-border bg-surface/40 shadow-card">
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-5 space-y-5">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center text-center px-4">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full border border-border bg-elevated text-xl">
              💬
            </div>
            <p className="font-display text-base font-medium text-ink">Ask anything about your document</p>
            <p className="mt-1 text-sm text-ink-muted max-w-xs">
              Answers are grounded in the retrieved context from your PDF.
            </p>

            <div className="mt-6 flex flex-wrap justify-center gap-2 max-w-lg">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => submit(s)}
                  className="rounded-full border border-border bg-elevated px-3.5 py-1.5 text-xs text-ink-muted transition-colors hover:border-accent-blue/40 hover:text-accent-blue"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) =>
            m.role === "user" ? (
              <UserBubble key={i} content={m.content} />
            ) : (
              <AssistantBubble key={i} message={m} />
            )
          )
        )}

        {isProcessing ? <LiveProcessing statuses={liveStatuses} stats={liveStats} /> : null}
      </div>

      <div className="border-t border-border p-4">
        <div className="flex items-end gap-2 rounded-xl border border-border bg-elevated px-3 py-2 focus-within:border-accent-blue/50 transition-colors">
          <textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Ask something about your PDF..."
            disabled={isProcessing}
            className="max-h-32 flex-1 resize-none bg-transparent py-1.5 text-sm text-ink placeholder:text-ink-faint outline-none disabled:opacity-60"
          />
          <button
            onClick={() => submit()}
            disabled={isProcessing || !input.trim()}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent-blue text-white transition-opacity disabled:opacity-30"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4">
              <path fill="currentColor" d="M3 12l18-9-6 9 6 9-18-9zm3.2 0L18 6l-9 6 9 6-11.8-6z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
