import { useRef, useState } from "react";

export default function UploadZone({ onFile, disabled }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  function validateAndSend(file) {
    if (!file) return;
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }
    setError("");
    onFile(file);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    validateAndSend(e.dataTransfer.files?.[0]);
  }

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`group relative flex flex-col items-center justify-center gap-4 rounded-2xl border border-dashed px-8 py-16 text-center transition-all duration-200
          ${disabled ? "cursor-not-allowed opacity-60 border-border" : "cursor-pointer"}
          ${dragging ? "border-accent-blue bg-accent-blue/[0.06] shadow-glow" : "border-border hover:border-accent-blue/50 hover:bg-white/[0.015]"}
        `}
      >
        <div
          className={`flex h-14 w-14 items-center justify-center rounded-full border transition-colors
          ${dragging ? "border-accent-blue/60 bg-accent-blue/10" : "border-border bg-elevated group-hover:border-accent-blue/40"}`}
        >
          <svg viewBox="0 0 24 24" className="h-6 w-6 text-accent-blue">
            <path
              fill="currentColor"
              d="M12 3l5 5h-3v6h-4V8H7l5-5zM5 19h14v2H5v-2z"
            />
          </svg>
        </div>

        <div>
          <p className="font-display text-lg font-medium text-ink">
            {dragging ? "Drop it here" : "Drop your PDF here"}
          </p>
          <p className="mt-1 text-sm text-ink-muted">or browse from your device</p>
        </div>

        <button
          type="button"
          disabled={disabled}
          onClick={(e) => {
            e.stopPropagation();
            inputRef.current?.click();
          }}
          className="mt-1 rounded-lg border border-accent-blue/30 bg-accent-blue/10 px-5 py-2 text-sm font-medium text-accent-blue transition-colors hover:bg-accent-blue/20 disabled:cursor-not-allowed"
        >
          Browse Files
        </button>

        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          disabled={disabled}
          onChange={(e) => validateAndSend(e.target.files?.[0])}
        />
      </div>

      {error ? <p className="mt-3 text-center text-sm text-accent-red">{error}</p> : null}
    </div>
  );
}
