// content.js – injected into every page by the browser.
//
// Creates a collapsible sidebar (Shadow DOM) on the right side of the page.
// Auto-analysis starts on load; results arrive via a message from background.js.
// The popup "Analyse" button still works and also updates the sidebar.

// ── Message listener ──────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.action === "getPageText") {
    sendResponse({ text: document.body?.innerText?.trim() ?? "", imageUrl: findMainImage() });
    return true;
  }
  if (message.action === "showResult") {
    if (!_shadow) createSidebar();
    setSidebarResult(message.result, message.error, message.imageNote);
  }
});

// ── Auto-analyze on load ──────────────────────────────────────────────────

function autoAnalyze() {
  const text = document.body?.innerText?.trim() ?? "";
  if (!text) return;

  createSidebar();
  setSidebarAnalyzing();

  try {
    chrome.runtime.sendMessage({ action: "autoAnalyze", text });
  } catch {
    setSidebarResult(null, "Extension context invalid. Try reloading the page.");
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", autoAnalyze);
} else {
  autoAnalyze();
}

// ── Sidebar core ──────────────────────────────────────────────────────────

let _shadow   = null;
let _expanded = false;

function createSidebar() {
  if (document.getElementById("__mmd_host__")) return;

  const host = document.createElement("div");
  host.id = "__mmd_host__";
  // Zero-size fixed host so the shadow DOM doesn't affect page layout at all
  Object.assign(host.style, {
    position:      "fixed",
    top:           "0",
    right:         "0",
    width:         "0",
    height:        "0",
    zIndex:        "2147483647",
    pointerEvents: "none",
  });

  _shadow = host.attachShadow({ mode: "open" });
  _shadow.innerHTML = SIDEBAR_TEMPLATE;
  _shadow.getElementById("mmd-toggle").addEventListener("click", () => setExpanded(!_expanded));

  document.body?.appendChild(host);
}

function setExpanded(open) {
  _expanded = open;
  const sidebar = _shadow?.getElementById("sidebar");
  const arrow   = _shadow?.querySelector(".mmd-arrow");
  if (!sidebar || !arrow) return;
  sidebar.classList.toggle("expanded", open);
  // ‹ = "pull open" (sidebar collapsed);  › = "push closed" (sidebar open)
  arrow.textContent = open ? "›" : "‹";
}

// ── Sidebar states ────────────────────────────────────────────────────────

function setSidebarAnalyzing() {
  if (!_shadow) return;
  _shadow.getElementById("mmd-strip").className = "mmd-strip";
  _shadow.getElementById("mmd-body").innerHTML = `
    <div class="mmd-state-row">
      <div class="mmd-spinner"></div>
      <span>Analyzing page…</span>
    </div>
    <ul class="mmd-steps">
      <li>Extracting page text</li>
      <li>Capturing screenshot</li>
      <li>Running model</li>
    </ul>
  `;
  setExpanded(true);
}

function setSidebarResult(result, error, imageNote) {
  if (!_shadow) return;

  if (error || !result) {
    _shadow.getElementById("mmd-strip").className = "mmd-strip error";
    _shadow.getElementById("mmd-body").innerHTML = `
      <div class="mmd-risk error">Error</div>
      <div class="mmd-error-msg">${error ?? "Unknown error"}</div>
    `;
    setExpanded(true);
    return;
  }

  const risk = result.risk || "UNKNOWN";
  const pct  = Math.round((result.confidence || 0) * 100);
  const exps = (result.explanations || []).map(e => `<li>${e}</li>`).join("");
  const note = result.used_image ? "screenshot + text" : "text only";

  _shadow.getElementById("mmd-strip").className = `mmd-strip ${risk.toLowerCase()}`;
  _shadow.getElementById("mmd-body").innerHTML = `
    <div class="mmd-risk ${risk}">${risk} RISK</div>
    <div class="mmd-pct">${pct}% probability of misinformation</div>
    <hr class="mmd-hr"/>
    ${exps ? `<div class="mmd-section">Key signals</div><ul class="mmd-exps">${exps}</ul>` : ""}
    ${imageNote ? `<div class="mmd-note">${imageNote}</div>` : ""}
    <div class="mmd-meta">${note}</div>
  `;
  setExpanded(true);
}

// ── Shadow DOM template ───────────────────────────────────────────────────

const SIDEBAR_TEMPLATE = `<style>
  /* All styles are scoped inside the shadow root */
  #sidebar {
    position: fixed;
    top: 0;
    right: 0;
    width: 300px;
    height: 100vh;
    background: #fff;
    border-left: 1px solid #e5e7eb;
    box-shadow: -3px 0 24px rgba(0,0,0,0.08);
    /* Collapsed: show only the 40px toggle tab */
    transform: translateX(calc(100% - 40px));
    transition: transform 0.22s cubic-bezier(.4,0,.2,1);
    display: flex;
    flex-direction: column;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
    line-height: 1.5;
    color: #111827;
    pointer-events: all;
    box-sizing: border-box;
  }
  #sidebar.expanded { transform: translateX(0); }
  #sidebar * { box-sizing: border-box; }

  /* Coloured strip across the top */
  .mmd-strip {
    height: 3px;
    flex-shrink: 0;
    background: #4f46e5;
    transition: background 0.3s;
  }
  .mmd-strip.high   { background: #dc2626; }
  .mmd-strip.medium { background: #d97706; }
  .mmd-strip.low    { background: #16a34a; }
  .mmd-strip.error  { background: #9ca3af; }

  /* Toggle tab — sits on the left edge of the sidebar */
  #mmd-toggle {
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 40px;
    height: 52px;
    background: #4f46e5;
    border: none;
    border-radius: 7px 0 0 7px;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 3px;
    color: #fff;
    padding: 0;
    box-shadow: -2px 0 8px rgba(0,0,0,0.14);
    transition: background 0.15s;
    z-index: 1;
  }
  #mmd-toggle:hover { background: #4338ca; }

  .mmd-arrow {
    font-size: 17px;
    line-height: 1;
    display: block;
  }
  .mmd-tag {
    font-size: 8px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.8;
    line-height: 1;
    display: block;
  }

  /* Scrollable content area; padding-left clears the toggle tab */
  .mmd-inner {
    flex: 1;
    overflow-y: auto;
    padding: 18px 16px 20px 52px;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  .mmd-header-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9ca3af;
    margin: 0 0 18px;
  }

  #mmd-body { flex: 1; display: flex; flex-direction: column; }

  /* ── Analyzing state ── */
  .mmd-state-row {
    display: flex;
    align-items: center;
    gap: 9px;
    color: #4f46e5;
    font-weight: 500;
    margin-bottom: 12px;
  }
  .mmd-spinner {
    width: 15px;
    height: 15px;
    border: 2px solid #e0e7ff;
    border-top-color: #4f46e5;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .mmd-steps {
    list-style: none;
    padding: 0 0 0 24px;
    margin: 0;
    font-size: 11px;
    color: #9ca3af;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .mmd-steps li::before { content: "— "; }

  /* ── Result state ── */
  .mmd-risk {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 0.03em;
    line-height: 1.1;
    margin: 0 0 3px;
  }
  .mmd-risk.HIGH   { color: #b91c1c; }
  .mmd-risk.MEDIUM { color: #92400e; }
  .mmd-risk.LOW    { color: #166534; }
  .mmd-risk.error  { color: #b91c1c; font-size: 17px; }

  .mmd-pct {
    font-size: 12px;
    color: #6b7280;
    margin: 0 0 14px;
  }

  .mmd-hr {
    border: none;
    border-top: 1px solid #f0f0f0;
    margin: 0 0 13px;
  }

  .mmd-section {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9ca3af;
    margin: 0 0 8px;
  }

  .mmd-exps {
    padding: 0 0 0 15px;
    margin: 0 0 16px;
    font-size: 12px;
    color: #374151;
    line-height: 1.6;
  }
  .mmd-exps li { margin-bottom: 5px; }

  .mmd-meta {
    font-size: 11px;
    color: #9ca3af;
    margin-top: auto;
    padding-top: 12px;
    border-top: 1px solid #f3f4f6;
  }

  .mmd-note {
    font-size: 11px;
    color: #9ca3af;
    font-style: italic;
    margin: 0 0 6px;
  }

  .mmd-error-msg {
    font-size: 12px;
    color: #6b7280;
    margin: 6px 0 0;
    line-height: 1.5;
  }
</style>

<div id="sidebar">
  <div class="mmd-strip" id="mmd-strip"></div>
  <button id="mmd-toggle" aria-label="Expand Misinformation Detector">
    <span class="mmd-arrow">‹</span>
    <span class="mmd-tag">MMD</span>
  </button>
  <div class="mmd-inner">
    <div class="mmd-header-label">Misinformation Detector</div>
    <div id="mmd-body"></div>
  </div>
</div>`;

// ── Helpers ───────────────────────────────────────────────────────────────

function findMainImage() {
  const imgs = Array.from(document.querySelectorAll("img"));
  if (!imgs.length) return null;
  const visible = imgs.filter(img =>
    img.src &&
    !img.src.startsWith("data:") &&
    img.offsetParent !== null &&
    img.offsetWidth > 0 &&
    img.offsetHeight > 0
  );
  if (!visible.length) return null;
  return visible.reduce((best, img) =>
    img.offsetWidth * img.offsetHeight > best.offsetWidth * best.offsetHeight ? img : best
  ).src;
}
