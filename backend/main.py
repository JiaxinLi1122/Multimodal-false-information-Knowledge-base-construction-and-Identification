# backend/main.py
#
# FastAPI server for the Multi-False Detector extension.
#
# Run with:
#   uvicorn main:app --reload
#
# The trained Text_Concat_Vision model is loaded once at startup (inside
# model_service.py).  The /analyze endpoint currently uses a text-only
# fallback because the Chrome extension does not yet send an image.
# When image support is added, swap predict_text_only() for predict().

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importing model_service triggers model loading – this happens once when
# uvicorn starts, not on every request.
from model_service import predict_text_only

app = FastAPI(title="Multi-False Detector API", version="0.2.0")

# Allow the Chrome extension (and local dev tools) to reach this API.
# In production, replace "*" with your actual extension origin or domain.
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
# Endpoints
# --------------------------------------------------------------------------- #

@app.post("/analyze", response_model=AnalyseResponse)
def analyze(request: AnalyseRequest) -> AnalyseResponse:
    """
    Analyse page text and return a misinformation risk level.

    Request body:  { "text": "..." }
    Response body: { "risk": "LOW|MEDIUM|HIGH", "reason": "..." }

    Currently delegates to predict_text_only() because the extension only
    sends text.  Switch to predict(text, image_tensor) once image capture
    is implemented in the extension.
    """
    result = predict_text_only(request.text)
    return AnalyseResponse(**result)
