// popup.js – runs inside the extension popup window.
//
// Flow:
//   1. User clicks "Analyse Current Page"
//   2. Identify the active browser tab
//   3. Ask content.js for page text                 → step: "Extracting page text"
//   4. captureVisibleTab → base64 JPEG screenshot   → step: "Capturing screenshot"
//   5. POST { text, image_data } to /analyze        → step: "Running model"
//   6. Render the returned result
//
// If captureVisibleTab fails the request is sent without image_data and the
// backend falls through to text-only mode.

const BACKEND_URL = "http://localhost:8000/analyze";

const analyseBtn       = document.getElementById("analyseBtn");
const resultDiv        = document.getElementById("result");
const statusText       = document.getElementById("status-text");
const loadingOutput    = document.getElementById("loading-output");
const analysisOutput   = document.getElementById("analysis-output");
const riskBadge        = document.getElementById("risk-badge");
const confidenceLine   = document.getElementById("confidence-line");
const explanationsList = document.getElementById("explanations-list");
const metaLine         = document.getElementById("meta-line");

const STEPS = ["step-1", "step-2", "step-3"];

// ── State helpers ──────────────────────────────────────────────────────────

function showIdle(msg = "Press the button to analyse this page.") {
  resultDiv.className = "";
  statusText.textContent = msg;
  statusText.style.display = "";
  loadingOutput.style.display = "none";
  analysisOutput.style.display = "none";
}

function showLoading() {
  resultDiv.className = "";
  statusText.style.display = "none";
  loadingOutput.style.display = "";
  analysisOutput.style.display = "none";
  // Reset all steps to pending
  STEPS.forEach(id => {
    const el = document.getElementById(id);
    el.className = "";
  });
}

// Mark steps[0..activeIdx-1] as done, steps[activeIdx] as active, rest pending.
function setStep(activeIdx) {
  STEPS.forEach((id, idx) => {
    const el = document.getElementById(id);
    if (idx < activeIdx)       el.className = "done";
    else if (idx === activeIdx) el.className = "active";
    else                        el.className = "";
  });
}

// imageCaptured: boolean – whether a screenshot was successfully taken
function renderResult({ risk, confidence, explanations, used_image }, charCount, imageCaptured) {
  // Mark all steps done before transitioning
  STEPS.forEach(id => { document.getElementById(id).className = "done"; });

  // Brief pause so the user sees all steps complete, then show result
  setTimeout(() => {
    resultDiv.className = risk;
    loadingOutput.style.display = "none";
    analysisOutput.style.display = "";

    riskBadge.textContent = `${risk} RISK`;
    riskBadge.className = risk;

    const pct = Math.round(confidence * 100);
    confidenceLine.textContent = `${pct}% probability of misinformation`;

    explanationsList.innerHTML = "";
    for (const point of (explanations ?? [])) {
      const li = document.createElement("li");
      li.textContent = point;
      explanationsList.appendChild(li);
    }

    const inputNote  = used_image   ? "screenshot + text" : "text only";
    const captureNote = imageCaptured ? "screenshot captured" : "no screenshot";
    metaLine.textContent =
      `Model input: ${inputNote} · ${captureNote} · ${charCount.toLocaleString()} chars`;
  }, 350);
}

function renderError(message) {
  showIdle(`Error: ${message}`);
}

// ── Main click handler ─────────────────────────────────────────────────────

// On popup open: display any cached auto-analysis result for the current tab
(async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) return;

    const key  = `tab_${tab.id}`;
    const data = await chrome.storage.local.get(key);
    const hit  = data[key];

    if (!hit?.result || Date.now() - hit.ts > 10 * 60 * 1000) return;

    renderResult(hit.result, hit.charCount || 0, hit.imageCaptured || false);
    analyseBtn.textContent = "Re-Analyse Page";
  } catch (_) { /* silently ignore */ }
})();

analyseBtn.addEventListener("click", async () => {
  analyseBtn.disabled = true;
  showLoading();
  setStep(0); // "Extracting content" active

  try {
    // Step 1 – identify the active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) { renderError("Could not identify the active tab."); return; }

    // Step 2 – ask content.js for text + largest visible image URL
    const pageResponse = await chrome.tabs.sendMessage(tab.id, { action: "getPageText" });
    if (!pageResponse?.text) { renderError("No text received from the page."); return; }

    const rawText = pageResponse.text;
    const text    = rawText.slice(0, 2000);

    setStep(1); // "Capturing screenshot" active

    // Step 2 – capture a JPEG screenshot of the visible tab area.
    // Requires the "activeTab" permission (already declared in manifest.json).
    // Strip the "data:image/jpeg;base64," prefix before sending.
    let imageData = null;
    try {
      const dataUrl = await chrome.tabs.captureVisibleTab(null, { format: "jpeg", quality: 70 });
      imageData = dataUrl.replace(/^data:image\/\w+;base64,/, "");
    } catch (captureErr) {
      console.warn("Screenshot capture failed:", captureErr);
      document.getElementById("step-2").textContent = "No image – text-only analysis";
    }

    setStep(2); // "Running model" active

    // Step 3 – POST to backend (5 s timeout)
    let apiResponse;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    try {
      const res = await fetch(BACKEND_URL, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ text, image_data: imageData }),
        signal:  controller.signal,
      });
      clearTimeout(timer);

      if (!res.ok) { renderError(`Backend returned status ${res.status}.`); return; }
      apiResponse = await res.json();
    } catch (fetchErr) {
      clearTimeout(timer);
      if (fetchErr.name === "AbortError") {
        renderError("Backend not available (request timed out after 5 s).");
      } else {
        renderError("Backend not available. Is it running on port 8000?");
      }
      return;
    }

    renderResult(apiResponse, text.length, !!imageData);

    // Mirror result to the page sidebar (fire-and-forget)
    chrome.tabs.sendMessage(tab.id, { action: "showResult", result: apiResponse }).catch(() => {});

  } catch (err) {
    renderError(err.message);
  } finally {
    analyseBtn.disabled = false;
  }
});
