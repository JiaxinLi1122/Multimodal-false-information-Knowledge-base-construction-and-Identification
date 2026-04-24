// background.js – service worker that handles auto-analysis triggered by content.js.
//
// Flow:
//   content.js  →  { action: "autoAnalyze", text }  →  background
//   background truncates text, captures screenshot, POSTs to /analyze,
//   then sends result (or a friendly error) back to the tab.

const BACKEND_URL = "http://localhost:8000/analyze";
const MAX_TEXT    = 2000;   // characters sent to the model
const TIMEOUT_MS  = 5000;   // abort fetch after 5 s

chrome.runtime.onMessage.addListener((message, sender) => {
  if (message.action !== "autoAnalyze") return;
  const tabId = sender.tab?.id;
  if (!tabId) return;
  runAnalysis(tabId, message.text);
});

async function runAnalysis(tabId, rawText) {
  // Truncate to keep the payload small and avoid model overload
  const text = rawText.slice(0, MAX_TEXT);

  // Screenshot – non-fatal; falls back to text-only if capture fails
  let imageData = null;
  let imageNote = null;
  try {
    const dataUrl = await chrome.tabs.captureVisibleTab(null, { format: "jpeg", quality: 70 });
    imageData = dataUrl.replace(/^data:image\/\w+;base64,/, "");
  } catch (e) {
    console.warn("[MMD] Screenshot capture failed:", e.message);
    imageNote = "No image found, using text-only analysis";
  }

  // API call with timeout
  let result, error;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    const res = await fetch(BACKEND_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text, image_data: imageData }),
      signal:  controller.signal,
    });
    clearTimeout(timer);

    if (!res.ok) throw new Error(`Backend returned ${res.status}`);
    result = await res.json();

    await chrome.storage.local.set({
      [`tab_${tabId}`]: { result, charCount: text.length, imageCaptured: !!imageData, ts: Date.now() },
    });
  } catch (e) {
    error = toFriendlyError(e);
  }

  try {
    await chrome.tabs.sendMessage(tabId, { action: "showResult", result, error, imageNote });
  } catch (e) {
    console.warn("[MMD] Could not deliver result to tab:", e.message);
  }
}

function toFriendlyError(e) {
  if (e.name === "AbortError")               return "Backend not available (request timed out after 5 s)";
  if (e.message?.startsWith("Backend returned")) return e.message; // preserve HTTP status
  return "Backend not available. Is the server running on port 8000?";
}
