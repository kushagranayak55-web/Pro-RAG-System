const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function handle(res) {
  if (!res.ok) {
    let detail = "Something went wrong.";
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON — keep the default message
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/api/upload`, {
    method: "POST",
    body: formData,
  });

  return handle(res);
}

export async function askQuestion(sessionId, question) {
  const res = await fetch(`${API_URL}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question }),
  });

  return handle(res);
}

export async function deleteSession(sessionId) {
  try {
    await fetch(`${API_URL}/api/session/${sessionId}`, { method: "DELETE" });
  } catch {
    // best-effort cleanup — a failed delete shouldn't block the UI
  }
}
