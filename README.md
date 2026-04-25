# Multi-False Detector

**A Chrome extension that checks whether the news you're reading is real or fake — automatically, as you browse.**

---

## Features

- Analyzes the current page the moment you open it — no button needed
- Reads both the article text **and** the page image together for a smarter result
- Shows a clear risk level: **LOW**, **MEDIUM**, or **HIGH**
- Gives you a short explanation of why the page was flagged
- Works on any news article or webpage

---

## Demo

> Screenshot / GIF coming soon

<!-- Replace this placeholder with a real screenshot or demo GIF -->
![Demo placeholder](docs/demo-placeholder.png)

---

## Installation (Manual)

The extension is not on the Chrome Web Store yet. Follow these steps to install it yourself.

**Step 1 — Download the project**

Click the green **Code** button on this page → **Download ZIP**, then unzip it anywhere on your computer.

**Step 2 — Open Chrome Extensions**

In Chrome, go to:
```
chrome://extensions/
```

**Step 3 — Enable Developer Mode**

In the top-right corner of that page, toggle **Developer mode** ON.

**Step 4 — Load the extension**

Click **Load unpacked**, then select the `extension` folder inside the unzipped project.

**Step 5 — Done**

The Multi-False Detector icon will appear in your Chrome toolbar. Pin it for easy access.

---

## How to Use

1. **Start the backend** (one-time setup — see below)
2. Open any news article or webpage in Chrome
3. The extension analyzes the page automatically
4. Click the extension icon to see the result:
   - The **risk level** (LOW / MEDIUM / HIGH)
   - A short **reason** explaining the result
   - Whether the page image was included in the analysis

That's it. No account, no sign-up.

---

## Starting the Backend

The extension needs a small local server running on your computer to do the analysis.

**Requirements:** Python 3.8 or newer

```bash
# Install dependencies (first time only)
cd backend
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload
```

The server starts at `http://localhost:8000`. Keep this terminal window open while you browse.

> The first startup takes 10–20 seconds while the AI model loads. After that, analysis is fast.

---

## Project Structure

```
extension/        Chrome extension files (load this folder in Chrome)
backend/          Local server that runs the detection model
data/             Training datasets (Twitter and Weibo news posts)
saved_models/     Trained model checkpoint
config/           Model and training settings
crawler/          Scripts used to collect training data
```

---

## Tech Stack

| Part | What it uses |
|------|-------------|
| Chrome Extension | JavaScript, Manifest V3 |
| Backend Server | Python, FastAPI |
| Text Analysis | BERT |
| Image Analysis | VGG-19 |
| Model Training | PyTorch |

---

## License

This project is for research and educational purposes only.
