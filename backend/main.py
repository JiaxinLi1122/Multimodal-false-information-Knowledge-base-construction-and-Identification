# backend/main.py
#
# FastAPI server for the Multi-False Detector extension.
#
# Run with:
#   uvicorn main:app --reload
#
# Current behaviour: keyword-based fake analysis (no real model).
# TODO: swap analyse_text() for a call to the BERT+VGG pipeline once the
#       model server is integrated.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Multi-False Detector API", version="0.1.0")

# Allow the Chrome extension (and local dev) to call this API.
# In production, restrict origins to your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #

class AnalyseRequest(BaseModel):
    text: str


class AnalyseResponse(BaseModel):
    risk: str   # "LOW" | "MEDIUM" | "HIGH"
    reason: str


# --------------------------------------------------------------------------- #
# Keyword rules – mirrors the logic in extension/popup.js so both surfaces
# agree until the real model is wired up.
# --------------------------------------------------------------------------- #

HIGH_KEYWORDS = [
    "shocking", "secret", "they don't want you to know", "wake up",
    "banned", "suppressed", "cover-up", "what they're hiding",
]

LOW_KEYWORDS = [
    "study", "research", "according to", "published", "evidence",
    "journal", "scientists", "peer-reviewed",
]


def analyse_text(text: str) -> AnalyseResponse:
    """Return a risk level and reason based on keyword matching."""
    lower = text.lower()

    # Check high-risk keywords first (highest priority)
    for kw in HIGH_KEYWORDS:
        if kw in lower:
            return AnalyseResponse(
                risk="HIGH",
                reason=f'Contains sensationalist language ("{kw}") commonly associated with misinformation.',
            )

    # Check low-risk keywords
    for kw in LOW_KEYWORDS:
        if kw in lower:
            return AnalyseResponse(
                risk="LOW",
                reason=f'Contains evidence-based language ("{kw}") typical of credible reporting.',
            )

    # Default: no strong signal either way
    return AnalyseResponse(
        risk="MEDIUM",
        reason="No strong indicators found. Manual review recommended.",
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.post("/analyze", response_model=AnalyseResponse)
def analyze(request: AnalyseRequest) -> AnalyseResponse:
    """
    Analyse a piece of text and return a misinformation risk level.

    Request body:  { "text": "..." }
    Response body: { "risk": "LOW|MEDIUM|HIGH", "reason": "..." }
    """
    return analyse_text(request.text)
