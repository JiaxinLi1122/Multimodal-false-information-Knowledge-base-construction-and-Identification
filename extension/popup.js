// popup.js – runs inside the extension popup window.
//
// Flow:
//   1. User clicks "Analyse Current Page"
//   2. We find the active browser tab
//   3. We send a message to content.js to get the visible page text
//   4. We POST the text to the local backend at /analyze
//   5. The backend returns { risk, reason } which we render here

const BACKEND_URL = "http://localhost:8000/analyze";

const analyseBtn = document.getElementById("analyseBtn");
const resultDiv  = document.getElementById("result");

// Maps risk level to a background colour for the result box
const LEVEL_COLOUR = { HIGH: "#fee2e2", MEDIUM: "#fef9c3", LOW: "#dcfce7" };

// Render a successful API response in the result box
function renderResult({ risk, reason }, charCount) {
  resultDiv.style.background = LEVEL_COLOUR[risk] ?? "#f3f4f6";
  resultDiv.textContent =
    `Risk level: ${risk}\n\n` +
    `${reason}\n\n` +
    `(${charCount.toLocaleString()} characters sent to backend)`;
}

// Render an error message in the result box
function renderError(message) {
  resultDiv.style.background = "#f3f4f6";
  resultDiv.textContent = `Error: ${message}`;
}

analyseBtn.addEventListener("click", async () => {
  analyseBtn.disabled = true;
  resultDiv.style.background = "#f3f4f6";
  resultDiv.textContent = "Extracting page text…";

  try {
    // Step 1 – identify the active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab?.id) {
      renderError("Could not identify the active tab.");
      return;
    }

    // Step 2 – ask content.js for the visible page text
    const pageResponse = await chrome.tabs.sendMessage(tab.id, { action: "getPageText" });

    if (!pageResponse?.text) {
      renderError("No text received from the page.");
      return;
    }

    // Step 3 – show loading state while the API call is in flight
    resultDiv.textContent = "Analyzing…";

    // Step 4 – POST text to the backend
    let apiResponse;
    try {
      const res = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: pageResponse.text }),
      });

      if (!res.ok) {
        renderError(`Backend returned status ${res.status}.`);
        return;
      }

      apiResponse = await res.json();
    } catch (_networkErr) {
      // fetch() itself throws when the server is unreachable
      renderError("Failed to connect to backend. Is it running on port 8000?");
      return;
    }

    // Step 5 – display the result
    renderResult(apiResponse, pageResponse.text.length);

  } catch (err) {
    renderError(err.message);
  } finally {
    analyseBtn.disabled = false;
  }
});
